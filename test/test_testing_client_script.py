# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context
from ag2.stream import MemoryStream
from ag2.testing import TestClient


@pytest.mark.asyncio()
async def test_exhausted_script_says_the_script_is_exhausted() -> None:
    """One model turn more than the script holds is reported as a short script.

    A ``StopIteration`` leaving the coroutine would be rewritten by PEP 479 into
    ``RuntimeError: coroutine raised StopIteration``, which names neither the test
    double nor the script — the caller cannot tell an over-running agent from a
    framework failure.
    """
    client = TestClient("only turn")
    context = Context(stream=MemoryStream())

    assert (await client([], context)).content == "only turn"

    with pytest.raises(RuntimeError, match=r"TestClient script exhausted after 1 turn"):
        await client([], context)
