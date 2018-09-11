from typing import TypeVar, Generic


T = TypeVar('T')


class MutableFixture(Generic[T]):
    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, value: T):
        self._value = value

    def __repr__(self):
        return repr(self._value)
