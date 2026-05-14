from __future__ import annotations

from pathlib import Path

from siteinsight.utils import load_json, save_json


class CreditStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        if not self.path.exists():
            save_json(self.path, {"balance": 25})

    def get_balance(self) -> int:
        data = load_json(self.path, {"balance": 25})
        return int(data.get("balance", 25))

    def set_balance(self, value: int) -> None:
        save_json(self.path, {"balance": max(0, int(value))})

    def add(self, amount: int) -> int:
        new_value = self.get_balance() + max(0, int(amount))
        self.set_balance(new_value)
        return new_value

    def consume(self, amount: int = 1) -> bool:
        balance = self.get_balance()
        if balance < amount:
            return False
        self.set_balance(balance - amount)
        return True