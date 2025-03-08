import os
import sys

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label

# 导入自定义模块
from scraper import scrape_and_save
from frequency import recommend_numbers
from randomForests import train_and_save_all_models, load_models_and_predict_next_draw
from lstm_predictor import train_lstm_model, predict_next_draw as lstm_predict_next
from xgb_predictor import train_xgb_models, predict_next_draw as xgb_predict_next_draw
from gan_predictor import train_gan_and_save, load_gan_and_generate_norepeat

# 默认字体（可根据需要修改）
default_font = "SimHei"


class MainApp(App):
    def build(self):
        # 启动时先爬取更新数据
        newest_date, added_count = scrape_and_save()

        # 根布局采用垂直的BoxLayout
        root_layout = BoxLayout(orientation="vertical", spacing=10, padding=10)

        # 第一行：预测期数输入框
        input_layout = BoxLayout(
            orientation="horizontal", size_hint=(1, None), height=40, spacing=10
        )
        lbl = Label(
            text="预测使用历史期数:",
            size_hint=(None, None),
            width=80,
            height=40,
            font_name=default_font,
        )
        self.n_recent_input = TextInput(
            text="100",
            multiline=False,
            input_filter="int",
            size_hint=(None, None),
            width=60,
            height=40,
            halign="center",
            font_name=default_font,
        )
        input_layout.add_widget(lbl)
        input_layout.add_widget(self.n_recent_input)
        root_layout.add_widget(input_layout)

        # 第二行：左右两个列，左列放预测按钮，右列放训练按钮
        columns_layout = BoxLayout(
            orientation="horizontal", spacing=20, size_hint=(1, None), height=250
        )

        # 左侧预测按钮的垂直布局
        pred_layout = BoxLayout(orientation="vertical", spacing=10)
        btn_freq = Button(
            text="频率预测", size_hint=(1, None), height=40, font_name=default_font
        )
        btn_freq.bind(on_press=lambda instance: self.on_frequency_predict())
        pred_layout.add_widget(btn_freq)

        btn_rf_predict = Button(
            text="RF预测", size_hint=(1, None), height=40, font_name=default_font
        )
        btn_rf_predict.bind(on_press=lambda instance: self.on_rf_predict())
        pred_layout.add_widget(btn_rf_predict)

        btn_lstm_predict = Button(
            text="LSTM预测", size_hint=(1, None), height=40, font_name=default_font
        )
        btn_lstm_predict.bind(on_press=lambda instance: self.on_lstm_predict())
        pred_layout.add_widget(btn_lstm_predict)

        btn_xgb_predict = Button(
            text="XGB预测", size_hint=(1, None), height=40, font_name=default_font
        )
        btn_xgb_predict.bind(on_press=lambda instance: self.on_xgb_predict())
        pred_layout.add_widget(btn_xgb_predict)

        btn_gan_generate = Button(
            text="GAN生成", size_hint=(1, None), height=40, font_name=default_font
        )
        btn_gan_generate.bind(on_press=lambda instance: self.on_gan_generate())
        pred_layout.add_widget(btn_gan_generate)

        # 右侧训练按钮的垂直布局
        train_layout = BoxLayout(orientation="vertical", spacing=10)
        btn_rf_train = Button(
            text="RF训练", size_hint=(1, None), height=40, font_name=default_font
        )
        btn_rf_train.bind(on_press=lambda instance: self.on_rf_train())
        train_layout.add_widget(btn_rf_train)

        btn_lstm_train = Button(
            text="LSTM训练", size_hint=(1, None), height=40, font_name=default_font
        )
        btn_lstm_train.bind(on_press=lambda instance: self.on_lstm_train())
        train_layout.add_widget(btn_lstm_train)

        btn_xgb_train = Button(
            text="XGB训练", size_hint=(1, None), height=40, font_name=default_font
        )
        btn_xgb_train.bind(on_press=lambda instance: self.on_xgb_train())
        train_layout.add_widget(btn_xgb_train)

        btn_gan_train = Button(
            text="GAN训练", size_hint=(1, None), height=40, font_name=default_font
        )
        btn_gan_train.bind(on_press=lambda instance: self.on_gan_train())
        train_layout.add_widget(btn_gan_train)

        # 将左右两列加入水平布局
        columns_layout.add_widget(pred_layout)
        columns_layout.add_widget(train_layout)
        root_layout.add_widget(columns_layout)

        # 第三部分：日志输出窗口，使用ScrollView包含一个只读的TextInput
        self.log_text = TextInput(
            text="", readonly=True, font_size=14, font_name=default_font
        )
        scroll = ScrollView(size_hint=(1, 1))
        scroll.add_widget(self.log_text)
        root_layout.add_widget(scroll)

        # 初始日志信息
        self.log_message(f"爬取完成: 最新={newest_date}, 新增={added_count}.")
        self.log_message("欢迎使用！")

        return root_layout

    def get_n_recent(self):
        """
        从预测期数输入框中获取期数，若无效则默认返回100。
        """
        try:
            num_str = self.n_recent_input.text.strip()
            val = int(num_str)
            return val if val > 0 else 100
        except:
            return 100

    def log_message(self, msg):
        """
        将日志追加到日志输出窗口，并滚动到底部。
        """
        self.log_text.text += msg + "\n"
        self.log_text.cursor = (0, len(self.log_text.text))
        self.log_text._refresh_text_from_property("text")

    def on_frequency_predict(self):
        try:
            n_recent = self.get_n_recent()
            reds, blue = recommend_numbers(csv_file="History.csv", n_recent=n_recent)
            self.log_message(f"[频率预测] 红球={reds}, 蓝球={blue}")
        except Exception as e:
            self.log_message(f"[频率预测出错]: {e}")

    def on_rf_predict(self):
        try:
            reds, blue = load_models_and_predict_next_draw("History.csv", "models")
            self.log_message(f"[RF预测] 红球={reds}, 蓝球={blue}")
        except Exception as e:
            self.log_message(f"[RF预测出错]: {e}")

    def on_rf_train(self):
        try:
            train_and_save_all_models(csv_file="History.csv", model_dir="models")
            self.log_message("[RF训练] 完成!")
        except Exception as e:
            self.log_message(f"[RF训练出错]: {e}")

    def on_lstm_predict(self):
        try:
            n_recent = self.get_n_recent()
            reds, blue = lstm_predict_next(
                "History.csv", "lstm_model.h5", window_size=n_recent
            )
            self.log_message(f"[LSTM预测] 红球={reds}, 蓝球={blue}")
        except Exception as e:
            self.log_message(f"[LSTM预测出错]: {e}")

    def on_lstm_train(self):
        try:
            n_recent = self.get_n_recent()
            train_lstm_model(
                "History.csv",
                "lstm_model.h5",
                window_size=n_recent,
                epochs=20,
                batch_size=16,
            )
            self.log_message("[LSTM训练] 完成!")
        except Exception as e:
            self.log_message(f"[LSTM训练出错]: {e}")

    def on_xgb_predict(self):
        try:
            reds, blue = xgb_predict_next_draw("History.csv", "xgb_models")
            self.log_message(f"[XGB预测] 红球={reds}, 蓝球={blue}")
        except Exception as e:
            self.log_message(f"[XGB预测出错]: {e}")

    def on_xgb_train(self):
        try:
            train_xgb_models("History.csv", "xgb_models")
            self.log_message("[XGB训练] 完成!")
        except Exception as e:
            self.log_message(f"[XGB训练出错]: {e}")

    def on_gan_generate(self):
        try:
            data = load_gan_and_generate_norepeat(
                "gan_models", "G.pth", num_samples=1, z_dim=16, max_attempts=50
            )
            reds = data[0, :6]
            blue = data[0, 6]
            self.log_message(f"[GAN生成] 红球={reds}, 蓝球={blue}")
        except Exception as e:
            self.log_message(f"[GAN生成出错]: {e}")

    def on_gan_train(self):
        try:
            train_gan_and_save(
                "History.csv",
                "gan_models",
                "G.pth",
                "D.pth",
                num_epochs=300,
                batch_size=32,
                z_dim=16,
            )
            self.log_message("[GAN训练] 完成!")
        except Exception as e:
            self.log_message(f"[GAN训练出错]: {e}")


if __name__ == "__main__":
    MainApp().run()
