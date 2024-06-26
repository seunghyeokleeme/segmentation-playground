import pandas as pd
import matplotlib.pyplot as plt

def plot_history(file_path):
    df = pd.read_csv(file_path)
    plt.figure(figsize=(12, 6))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['loss'], color='blue', label='train_loss')
    plt.plot(df['epoch'], df['val_loss'], color='red', label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['accuracy'], color='blue', label='train_accuracy')
    plt.plot(df['epoch'], df['val_accuracy'], color='red', label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()

    plt.show()

# 그래프 출력