import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def visualize_neural(neural_array, title=f'Neural Array', neural_channel=10, offset=1):
    title = title + '- channel {}'.format(neural_channel)
    fig = plt.figure()
    for i in range(len(neural_array)):
        plt.plot(neural_array[i][neural_channel] + offset*i)
        plt.title(title)
    return fig


def visualize_audio(audio_motifs, title=f'Audio Motifs', offset=40000):
    fig = plt.figure()
    for i in range(len(audio_motifs)):
        plt.plot(audio_motifs[i] + offset*i)
        plt.title(title)
    return fig


def visualize_audio_embeddings(audio_embeddings, title=f'Audio Embeddings', embedding_dim=0, offset=10):
    title = title + '- embed dim {}'.format(embedding_dim)
    fig = plt.figure()
    for i in range(len(audio_embeddings)):
        plt.plot(audio_embeddings[i][embedding_dim] + offset*i)
        plt.title(title)
    return fig


def visualize_loss_error(train_loss_error, test_loss_error=None, plot_epochs=-1, title=''):
    fig, ax = plt.subplots()
    ax.plot(train_loss_error[:plot_epochs], label='train')
    if test_loss_error:
        ax.plot(test_loss_error[:plot_epochs], label='val')
    ax.set_title(title)
    ax.legend()
    return fig

