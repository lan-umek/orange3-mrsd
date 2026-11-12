"""Small GUI helpers shared by the widgets."""
from AnyQt.QtCore import QEvent, QObject, Qt, QTimer
from AnyQt.QtWidgets import QAbstractSpinBox, QComboBox, QSlider


class _WheelGuard(QObject):
    """Combo boxes, spin boxes and sliders must not change their value when the
    mouse wheel passes over them while the panel is being scrolled: the wheel
    event is ignored (so the enclosing scroll area scrolls instead).  Values
    change only by clicking or with the keyboard."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            event.ignore()
            return True
        return False


def guard_wheel(widget):
    """Install the wheel guard on every combo box / spin box / slider under `widget`."""
    guard = getattr(widget, "_wheel_guard", None)
    if guard is None:
        guard = widget._wheel_guard = _WheelGuard(widget)
    for cls in (QComboBox, QAbstractSpinBox, QSlider):
        for child in widget.findChildren(cls):
            child.setFocusPolicy(Qt.StrongFocus)
            child.installEventFilter(guard)


def guard_wheel_later(widget):
    """Call guard_wheel once the widget's constructor has finished."""
    QTimer.singleShot(0, lambda: guard_wheel(widget))
