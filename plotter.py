from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torchvision.transforms as trans
from utils import *

plt.rcParams['figure.figsize'] = (60.0, 60.0)


def plot_pqd_signals(file_name, fig_title, pqd_signals, labels_order, num_of_samples, fs):
    """
    Plot a set of pqd signals with their labels

    Args:
        file_name: name of the file
        fig_title: title of the figure
        pqd_signals: signals of pqd, measured in voltage[p.u.]
        labels_order: list of the labels with their names
        num_of_samples: number of samples in each pqd signal
        fs: sampling frequency of the pqd signal
    """
    path = os.getcwd() + '\\figures\\' + file_name + '.png'
    plt.figure(constrained_layout=True)
    plt.suptitle(fig_title, fontsize=100)
    time = np.arange(0, num_of_samples) / fs

    for i, label in enumerate(labels_order):
        plt.subplot(4, 4, i+1)
        plt.grid(True)
        plt.plot(time, pqd_signals[i])
        plt.title(label + ' signal', fontsize=60)
        plt.xlabel('Time[s]', fontsize=50)
        plt.xticks(fontsize=30)
        plt.ylabel('Voltage[p.u.]', fontsize=50)
        plt.yticks(fontsize=30)

    plt.savefig(path)
    plt.show()


def plot_cwt_signals(file_name, fig_title, model, cwt_signals, labels_order, le, epsilon=0,
                     num_of_samples=6400, fs=32000, f_min=100, f_max=2000):
    """
    Plot a set of images of pqd signals, after cwt type transform,
    which are in their original state or after an attack

    Args:
        file_name: name of the file
        fig_title: title of the figure
        model: PQD classifier
        cwt_signals: set of cwt signals
        labels_order: list of the labels' names
        le: encoder transform for converting labels from numbers to strings
        epsilon: epsilon used for the adversarial attack
    """

    path = os.getcwd() + '\\figures\\' + file_name + '.png'
    transform = trans.ToPILImage()
    plt.figure(constrained_layout=True)
    plt.suptitle(fig_title, fontsize=80)
    model.eval()

    with torch.no_grad():
        for i, label in enumerate(labels_order):
            ax = plt.subplot(4, 4, i+1)
            plt.grid(True)
            img = np.array(transform(cwt_signals[i]))/255
            output = model(cwt_signals[i].unsqueeze(0))
            prob, prediction = torch.max(output, dim=1)
            im = ax.imshow(img, cmap='jet', extent=[0, num_of_samples/fs, f_min, f_max])
            ax.set_aspect((num_of_samples/fs)/(f_max-f_min))
            plt.title('Signal: ' + label + '\nprediction: ' + str(le.inverse_transform(prediction)[0]) +
                      '\nconfidence: ' + f"{prob.cpu().numpy()[0]*100:.2f}%", fontsize=60)
            plt.xlabel('Time[s]', fontsize=50)
            plt.xticks(fontsize=30)
            plt.ylabel('Frequency[Hz]', fontsize=50)
            plt.yticks(fontsize=30)
            plt.colorbar(im)

    plt.savefig(path)
    plt.show()


def subplotter(epsilons, perf_met_1, perf_met_2, labels, marker, ylabel, title,
               perf_met_3=None, colors=None, xlabel="$\epsilon$"):
    """
    Make a subplot of 3 performance metric vectors, for different attacks or modes.

    Args:
        epsilons: epsilon values used for evaluation process
        perf_met_1: performance metric vector 1 - accuracy, f1 score, misclassification rate, or average robustness
        perf_met_2: performance metric vector 2
        labels: type of adversarial attack (FGSM, BIM, PGD, or DeepFool), or attacking mode (white, gray, or black box)
        marker: marker's type for plot
        ylabel: title for horizontal axis
        title: title of the subplot graph
        perf_met_3: performance metric vector 3
        colors: colors for the graphs inside the current subplots
        xlabel: title for the vertical axis

    """

    plt.grid(True)
    plt.plot(epsilons, perf_met_1, color=colors[0], label=labels[0], marker=marker)
    plt.plot(epsilons, perf_met_2, color=colors[1], label=labels[1], marker=marker)
    if perf_met_3 is not None:
        plt.plot(epsilons, perf_met_3, color=colors[2], label=labels[2], marker=marker)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(fontsize=20)
    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    plt.title(title, fontsize=40)


def eps_performance_plotter(model_name, legend_type, mode='', attack_type='', retrain_model_name=''):
    """
    Plot performance metrics results as function of epsilon, for the different attacks and modes:
    FGSM, BIM, PGD attacks and black/gray/white box modes.

    Args:
        model_name: PQD classifier's name
        legend_type: indication of the graphs in each subplot - either mode or attack
        mode: attack mode - black box, gray box, or white box
        attack_type: type of adversarial attack - FGSM, BIM, PGD, or DeepFool
        retrain_model_name: name of the PQD classifier after adversarial training process
    """

    epsilons = np.array([0, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1])
    colors = ['r', 'g', 'b']
    plt.rcParams['figure.figsize'] = (20.0, 20.0)
    plt.figure(constrained_layout=True)

    match legend_type:
        case 'attack':
            plt.suptitle("Performance metrics results of " + mode, fontsize=50)

            # Load the performance results vectors, for a fixed mode and attack dependence
            acc_fgsm, f1_fgsm, mcr_fgsm, avg_rob_fgsm = performance_vec_loader(model_name, mode, 'FGSM')
            acc_bim, f1_bim, mcr_bim, avg_rob_bim = performance_vec_loader(model_name, mode, 'BIM')
            acc_pgd, f1_pgd, mcr_pgd, avg_rob_pgd = performance_vec_loader(model_name, mode, 'PGD')

            plt.subplot(2, 2, 1)  # Accuracy Vs. epsilon graph
            plt.grid(True)
            subplotter(epsilons, acc_fgsm, acc_bim, ['FGSM', 'BIM', 'PGD'], 's', "Accuracy",
                       "Accuracy vs. " + "$\epsilon$", perf_met_3=acc_pgd, colors=colors)

            plt.subplot(2, 2, 2)  # F1 score Vs. epsilon graph
            plt.grid(True)
            subplotter(epsilons, f1_fgsm, f1_bim, ['FGSM', 'BIM', 'PGD'], 'D', "F1 score",
                       "F1 score vs. " + "$\epsilon$", perf_met_3=f1_pgd, colors=colors)

            plt.subplot(2, 2, 3)  # Misclassification rate Vs. epsilon graph
            plt.grid(True)
            subplotter(epsilons, mcr_fgsm, mcr_bim, ['FGSM', 'BIM', 'PGD'], '*', "Misclassification rate",
                       "Misclassification rate vs. " + "$\epsilon$", perf_met_3=mcr_pgd, colors=colors)

            plt.subplot(2, 2, 4)  # Average robustness Vs. epsilon graph
            plt.grid(True)
            subplotter(epsilons, avg_rob_fgsm, avg_rob_bim, ['FGSM', 'BIM', 'PGD'], 'o', r"$\hat{\rho}_{adv}$",
                       "Average robustness vs. " + "$\epsilon$", perf_met_3=avg_rob_pgd, colors=colors)

            file_name = model_name + '_' + mode + '_' + 'performance results'
            path = os.getcwd() + '\\figures\\' + file_name + '.png'
            plt.savefig(path)
            plt.show()

        case 'mode':
            plt.suptitle("Performance metrics results for different modes of " + attack_type, fontsize=50)

            # Load the performance results vectors, for a fixed attack and mode dependence
            acc_white, f1_white, mcr_white, avg_rob_white = performance_vec_loader(model_name, 'white box', attack_type)
            acc_gray, f1_gray, mcr_gray, avg_rob_gray = performance_vec_loader(model_name, 'gray box', attack_type)
            acc_black, f1_black, mcr_black, avg_rob_black = performance_vec_loader(model_name, 'black box', attack_type)

            plt.subplot(2, 2, 1)  # Accuracy Vs. epsilon graph
            plt.grid(True)
            subplotter(epsilons, acc_white, acc_gray, ['White box', 'Gray box', 'Black box'], 's',
                       "Accuracy", "Accuracy vs. " + "$\epsilon$", perf_met_3=acc_black, colors=colors)

            plt.subplot(2, 2, 2)  # F1 score Vs. epsilon graph
            plt.grid(True)
            subplotter(epsilons, f1_white, f1_gray, ['White box', 'Gray box', 'Black box'], 'D', "F1 score",
                       "F1 score vs. " + "$\epsilon$", perf_met_3=f1_black, colors=colors)

            plt.subplot(2, 2, 3)  # Misclassification rate Vs. epsilon graph
            plt.grid(True)
            subplotter(epsilons, mcr_white, mcr_gray, ['White box', 'Gray box', 'Black box'], '*',
                       "Misclassification rate", "Misclassification rate vs. " + "$\epsilon$",
                       perf_met_3=mcr_black, colors=colors)

            plt.subplot(2, 2, 4)  # Average robustness score Vs. epsilon graph
            plt.grid(True)
            subplotter(epsilons, avg_rob_white, avg_rob_gray, ['White box', 'Gray box', 'Black box'], 'o',
                       r"$\hat{\rho}_{adv}$", "Average robustness vs. " + "$\epsilon$",
                       perf_met_3=avg_rob_black, colors=colors)

            file_name = model_name + '_' + attack_type + '_' + 'performance results'
            path = os.getcwd() + '\\figures\\' + file_name + '.png'
            plt.savefig(path)
            plt.show()

        case 'compare':
            colors = ['r', 'g']
            plt.suptitle("Adversarial training - performance comparison", fontsize=50)

            # Load the performance results vectors, for a fixed attack and mode dependence, before and after retraining
            acc_org, f1_org, mcr_org, avg_rob_org = performance_vec_loader(model_name, mode, attack_type)
            acc_retrain, f1_retrain, mcr_retrain, avg_rob_retrain = performance_vec_loader(retrain_model_name, mode, attack_type)

            plt.subplot(2, 2, 1)  # Accuracy Vs. epsilon graph
            plt.grid(True)
            subplotter(epsilons, acc_org, acc_retrain, ['Original setup', 'Adversarial training setup'], 's',
                       "Accuracy", "Accuracy vs. " + "$\epsilon$", colors=colors)

            plt.subplot(2, 2, 2)  # F1 score Vs. epsilon graph
            plt.grid(True)
            subplotter(epsilons, f1_org, f1_retrain, ['Original setup', 'Adversarial training setup'], 'D', "F1 score",
                       "F1 score vs. " + "$\epsilon$", colors=colors)

            plt.subplot(2, 2, 3)  # Misclassification rate Vs. epsilon graph
            plt.grid(True)
            subplotter(epsilons, mcr_org, mcr_retrain, ['Original setup', 'Adversarial training setup'], '*',
                       "Misclassification rate", "Misclassification rate vs. " + "$\epsilon$", colors=colors)

            plt.subplot(2, 2, 4)  # Average robustness score Vs. epsilon graph
            plt.grid(True)
            subplotter(epsilons, avg_rob_org, avg_rob_retrain, ['Original setup', 'Adversarial training setup'],
                       'o', r"$\hat{\rho}_{adv}$", "Average robustness vs. " + "$\epsilon$", colors=colors)

            file_name = model_name + '_' + mode + '_' + attack_type + '_' + 'before and after adv training'
            path = os.getcwd() + '\\figures\\' + file_name + '.png'
            plt.savefig(path)
            plt.show()


def plot_confusion_matrix(cm, name, le, attack_type='', mode=''):
    """
    Plot confusion matrix of a certain PQD classifier

    Args:
        cm: confusion matrix
        name: type of the PQD classifier
        le: encoder transform for converting labels from numbers to strings
        attack_type: type of the adversarial attack
        mode: type of the attacking mode
    """
    # Define path to save figure
    path = os.getcwd() + '\\figures\\' + 'confusion_matrix_' + name + '_' + attack_type + '_' + mode + '.png'

    # Plot confusion matrix with custom colormap
    cmp = ConfusionMatrixDisplay(cm, display_labels=le.inverse_transform(np.arange(16)))
    fig, ax = plt.subplots(figsize=(8, 7), tight_layout=True)
    plt.grid(False)
    cmp.plot(include_values=True, cmap='Reds', ax=ax, xticks_rotation='vertical', colorbar=True)

    # Save and plot figure
    plt.savefig(path)
    plt.show()
