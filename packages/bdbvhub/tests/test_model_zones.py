"""The renewal models read the hub's zone data."""

from __future__ import annotations

import pytest

from bdbvhub.models import get_model


@pytest.mark.parametrize("name", ["renewal", "renewalsp", "renewalbvd"])
def test_renewal_models_read_zones(name: str) -> None:
    assert getattr(get_model(name), "reads_zones", False)
