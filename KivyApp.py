from kivy.app import App
from MainLayout import MainLayout


class KivyApp(App):
    def build(self):
        return MainLayout()


if __name__ == "__main__":
    KivyApp().run()
