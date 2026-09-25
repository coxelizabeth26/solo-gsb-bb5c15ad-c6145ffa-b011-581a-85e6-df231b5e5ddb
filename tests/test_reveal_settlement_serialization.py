"""揭示（单条 / 批量）与永久结算按同一事件锁串行化的并发回归测试。

缺陷场景：单条或批量揭示在 reveal 截止前通过阶段检查、尚未提交时，截止
后的结算抢先取得事件行锁，把该承诺计为未揭示并固化；随后揭示仍成功落库
——固化计数与 reveals 表从此矛盾。

修复约定：单条揭示、批量揭示与永久结算在入口先取得**同一把事件行锁**
（app.main.acquire_event_lock），取锁成功后才读取**唯一一次**服务端 UTC，
该时刻同时用于阶段判定与 created_at。因此只有两种串行次序：

  1. 结算先取得锁：等待锁的揭示在取锁后才读时钟，此刻已进入 closed，
     按 REVEAL_PHASE_CLOSED 整请求拒绝且不落库（单条 / 批量各一例）；
  2. 揭示先取得锁：全部揭示随事务提交后结算才取得锁并统计，固化计数
     必然包含这些先提交的合法揭示（单条 / 批量各一例）。

SQLite 不支持 SELECT ... FOR UPDATE（退化为普通 SELECT），无法在数据库
层真正阻塞，因此这里在锁边界 acquire_event_lock 上注入确定性的竞争者：

  - 「结算先完成」：竞争者在揭示请求取锁的同一调用点先同步完成结算，
    并把模拟时钟推进到揭示截止时刻，再把事件行返回给揭示请求；
  - 「揭示先完成」：竞争者（独立线程）在揭示取锁的同一调用点发起结算，
    但阻塞到揭示事务提交之后才放行，模拟它在事件锁上等待、揭示提交后
    才取得锁。
"""

import os
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone

_tmpdir = tempfile.mkdtemp(prefix="sealed_predictions_reveal_settle_lock_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/reveal_settle_lock.db"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.clock as clock_module  # noqa: E402
import app.main as main_module  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.hashing import compute_commitment  # noqa: E402
from app.schemas import SettlementCreate  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

T0 = datetime(2030, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
COMMIT_DL = T0 + timedelta(hours=1)
REVEAL_DL = T0 + timedelta(hours=2)
EPS = timedelta(microseconds=1)


class MutableClock:
    def __init__(self, start: datetime):
        self.t = start

    def __call__(self) -> datetime:
        return self.t


clock = MutableClock(T0)


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch):
    clock.t = T0
    monkeypatch.setattr(clock_module, "now_utc", clock)
    with client:  # 触发 lifespan 建表（幂等）
        yield


client = TestClient(main_module.app)


def text(i: int) -> str:
    return f"prediction-{i}"


def salt(i: int) -> str:
    return f"salt-{i}"


def h(i: int) -> str:
    return compute_commitment(text(i), salt(i))


def parse_ts(raw: str) -> datetime:
    text_value = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    return datetime.fromisoformat(text_value)


def reveal_item(i: int) -> dict:
    return {"original_text": text(i), "salt": salt(i)}


@pytest.fixture
def seeded_event():
    """创建事件并提交 3 个承诺（h1..h3），时间停在 commit 阶段。"""
    r = client.post("/events", json={
        "title": "揭示结算串行化事件",
        "commit_deadline": COMMIT_DL.isoformat(),
        "reveal_deadline": REVEAL_DL.isoformat(),
    })
    assert r.status_code == 201, r.text
    event_id = r.json()["id"]
    r = client.post(
        f"/events/{event_id}/commitments/batch",
        json={"commitment_hashes": [h(1), h(2), h(3)]},
    )
    assert r.status_code == 201, r.text
    return event_id


def settle_directly(event_id: str, standard_result: str = None) -> None:
    """用独立会话直接执行结算路由（绕过 HTTP 客户端，便于在锁注入点调用）。"""
    db = SessionLocal()
    try:
        main_module.settle_event(
            uuid.UUID(event_id),
            SettlementCreate(standard_result=standard_result or text(1)),
            db=db,
        )
    finally:
        db.close()


def install_settlement_first_racer(monkeypatch, event_id: str):
    """在揭示请求取锁点注入「先完成结算」的竞争者。

    揭示请求调用 acquire_event_lock 时：竞争者先以 closed 阶段时刻
    （REVEAL_DL）同步完成结算，随后再让揭示请求取得事件行。揭示请求
    取锁后读到的时钟已是 REVEAL_DL，必须按 REVEAL_PHASE_CLOSED 拒绝。
    """
    real_lock = main_module.acquire_event_lock
    state = {"in_racer": False}

    def racing_lock(db, lock_event_id):
        if state["in_racer"] or str(lock_event_id) != event_id:
            return real_lock(db, lock_event_id)
        # 模拟截止时刻到达、结算先取得锁并完成
        clock.t = REVEAL_DL
        state["in_racer"] = True
        try:
            settle_directly(event_id)
        finally:
            state["in_racer"] = False
        # 再把锁交给已等待到截止后的揭示请求
        return real_lock(db, lock_event_id)

    monkeypatch.setattr(main_module, "acquire_event_lock", racing_lock)


# --------------------------------------------------------------------------- #
# 次序 1：结算先完成 -> 后到揭示按阶段拒绝且不落库
# --------------------------------------------------------------------------- #

def test_single_reveal_after_settlement_won_lock_is_rejected_and_not_persisted(
    seeded_event, monkeypatch
):
    event_id = seeded_event
    install_settlement_first_racer(monkeypatch, event_id)

    # 揭示请求在截止前发起；竞争者在取锁点把时钟推进到 REVEAL_DL 并先结算
    clock.t = REVEAL_DL - EPS
    r = client.post(f"/events/{event_id}/reveals", json=reveal_item(1))
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["code"] == "REVEAL_PHASE_CLOSED"

    # 拒绝不落库：只有结算（固化该承诺为未揭示），没有揭示记录
    detail = client.get(f"/events/{event_id}").json()
    assert detail["reveal_count"] == 0
    settlement = client.get(f"/events/{event_id}/settlement").json()
    assert settlement["status"] == "settled"
    assert settlement["unrevealed_count"] == 3
    assert settlement["correct_count"] == 0


def test_batch_reveals_after_settlement_won_lock_fully_rolled_back(
    seeded_event, monkeypatch
):
    event_id = seeded_event
    install_settlement_first_racer(monkeypatch, event_id)

    clock.t = REVEAL_DL - EPS
    r = client.post(
        f"/events/{event_id}/reveals/batch",
        json={"reveals": [reveal_item(1), reveal_item(2)]},
    )
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["code"] == "REVEAL_PHASE_CLOSED"

    # 整批不落库；结算已把全部承诺固化为未揭示
    assert client.get(f"/events/{event_id}").json()["reveal_count"] == 0
    settlement = client.get(f"/events/{event_id}/settlement").json()
    assert settlement["status"] == "settled"
    assert settlement["unrevealed_count"] == 3
    assert settlement["correct_count"] == 0
    # 结算只确认一次：随后再次结算仍被永久确认冲突拒绝
    r = client.post(
        f"/events/{event_id}/settlement",
        json={"standard_result": text(2)},
    )
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "SETTLEMENT_ALREADY_CONFIRMED"


# --------------------------------------------------------------------------- #
# 次序 2：揭示先提交 -> 后取得锁的结算统计先提交的合法揭示
# --------------------------------------------------------------------------- #

def install_reveal_first_barrier(monkeypatch, event_id: str):
    """在揭示取锁点发起一个等待揭示提交后才放行的结算竞争者。

    竞争者在独立线程中启动（模拟其已请求事件行锁、正在锁上等待），但
    阻塞到揭示事务提交完成后才执行结算；提交钩子保证放行严格发生在
    揭示 commit 之后。
    """
    real_lock = main_module.acquire_event_lock
    reveal_committed = threading.Event()
    settlement_done = threading.Event()
    racer_thread_ids: set[int] = set()

    def racer():
        racer_thread_ids.add(threading.get_ident())
        # 等待揭示事务先提交（模拟在事件行锁上排队）
        assert reveal_committed.wait(timeout=5)
        # 取得锁时真实时间已越过揭示截止：结算此刻读时钟进入 closed
        clock.t = REVEAL_DL
        try:
            settle_directly(event_id)
        finally:
            settlement_done.set()

    def racing_lock(db, lock_event_id):
        if (
            threading.get_ident() in racer_thread_ids
            or str(lock_event_id) != event_id
        ):
            return real_lock(db, lock_event_id)
        # 启动在锁上排队的结算竞争者后立即放行揭示请求：竞争者阻塞到
        # 揭示事务提交（commit 钩子）后才执行，不在这里 join（否则会
        # 形成「等揭示提交 / 等锁返回」的自死锁）；测试通过 settlement_done
        # 事件等待竞争者结算完成
        t = threading.Thread(target=racer)
        t.start()
        return real_lock(db, lock_event_id)

    real_commit = Session.commit

    def commit_and_signal(self, *args, **kwargs):
        # SQLAlchemy 提交后会清空 session.new，须在提交前捕获待提交的揭示
        pending_reveal = any(
            isinstance(obj, main_module.Reveal) for obj in getattr(self, "new", [])
        )
        result = real_commit(self, *args, **kwargs)
        if pending_reveal:
            reveal_committed.set()
        return result

    monkeypatch.setattr(main_module, "acquire_event_lock", racing_lock)
    monkeypatch.setattr(Session, "commit", commit_and_signal)
    return settlement_done


def test_settlement_waiting_on_lock_counts_first_committed_single_reveal(
    seeded_event, monkeypatch
):
    event_id = seeded_event
    clock.t = REVEAL_DL - EPS  # 揭示期最后一刻：揭示合法
    done = install_reveal_first_barrier(monkeypatch, event_id)

    r = client.post(f"/events/{event_id}/reveals", json=reveal_item(1))
    assert r.status_code == 201, r.text
    assert parse_ts(r.json()["revealed_at"]) == REVEAL_DL - EPS

    assert done.wait(timeout=5)
    # 结算在揭示提交后才统计：h1 的揭示已落库，计入正确，其余两个未揭示
    settlement = client.get(f"/events/{event_id}/settlement").json()
    assert settlement["status"] == "settled"
    assert settlement["correct_count"] == 1
    assert settlement["incorrect_count"] == 0
    assert settlement["unrevealed_count"] == 2
    assert parse_ts(settlement["settled_at"]) == REVEAL_DL

    # 存库记录与固化计数一致，事件计数与结算之和吻合
    detail = client.get(f"/events/{event_id}").json()
    assert detail["reveal_count"] == 1
    assert (
        settlement["correct_count"]
        + settlement["incorrect_count"]
        + settlement["unrevealed_count"]
    ) == detail["commitment_count"]


def test_settlement_waiting_on_lock_counts_first_committed_batch_reveals(
    seeded_event, monkeypatch
):
    event_id = seeded_event
    clock.t = REVEAL_DL - EPS  # 揭示期最后一刻：整批合法
    done = install_reveal_first_barrier(monkeypatch, event_id)

    r = client.post(
        f"/events/{event_id}/reveals/batch",
        json={"reveals": [reveal_item(1), reveal_item(2)]},
    )
    assert r.status_code == 201, r.text
    assert {parse_ts(x["revealed_at"]) for x in r.json()["reveals"]} == {
        REVEAL_DL - EPS
    }

    assert done.wait(timeout=5)
    # 整批先随事务提交，结算后统计：h1 正确、h2 错误、h3 未揭示
    settlement = client.get(f"/events/{event_id}/settlement").json()
    assert settlement["status"] == "settled"
    assert settlement["correct_count"] == 1
    assert settlement["incorrect_count"] == 1
    assert settlement["unrevealed_count"] == 1

    detail = client.get(f"/events/{event_id}").json()
    assert detail["reveal_count"] == 2
    assert (
        settlement["correct_count"]
        + settlement["incorrect_count"]
        + settlement["unrevealed_count"]
    ) == detail["commitment_count"]
