from gui.controller import AppController
from gui.view import AppView


def main():
    controller = AppController(view=None)
    view = AppView(controller=controller)
    controller.view = view
    view.mainloop()


if __name__ == "__main__":
    main()
