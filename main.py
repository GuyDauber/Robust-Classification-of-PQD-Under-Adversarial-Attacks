import os
import numpy as np
import random
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import trainer
from models import *
from utils import *
from evaluater import *
from plotter import *
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import normalize
from attacker import *


def train_pqd_classifier(inputs, labels, model, name, device, mode):
    """
        Prepare a dataset for a pqd classification training process, and feed it into the training model block.

        Args:
            inputs: PQD signals
            labels: labels of the PQD signals
            model: DL-based PQD classifier
            name: string which indicates the architecture of the classifier
            device: cuda or cpu
            mode: training mode - white box, gray box, or black box
        """
    seed = 0
    train_set, val_set, _, _, _, train_set_adv, val_set_adv, _ = split_train_val_test(inputs, labels, seed=seed)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
    adv_train_loader = DataLoader(train_set_adv, batch_size=16, shuffle=True)
    adv_val_loader = DataLoader(val_set_adv, batch_size=16, shuffle=True)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    name += '_' + mode
    if mode == 'white box':
        trainer.train_model(model, name, train_loader, val_loader, optimizer, criterion,
                            n_epochs=100, patience=20, device=device)

    # adversary loaders for black and gray box
    trainer.train_model(model, name, adv_train_loader, adv_val_loader, optimizer, criterion,
                        n_epochs=100, patience=20, device=device)


def processing_stage(stage, cwt_signals_set, labels, labels_order, le, main_model_name, test_model_name,
                     num_of_samples=6400, fs=32e3, device='cpu', mat_name='dataset_Guy',
                     mode='', attack_type='', best_eps=None):
    """
        Sends the data to the desired stage in the process, and executes

        Args:
            stage: process's stage- plot, samples_test, train, evaluate, attack
            cwt_signals_set: signals of pqd, after the cwt type transform
            labels: labels of the PQD signals
            labels_order: list of the labels' names
            le: encoder transform for converting labels from numbers to strings
            main_model_name: model of the DL-based PQD classifier, used for training and making the adversarial attacks
            test_model_name: model of the DL-based PQD classifier, used for evaluation of the process
            device: cuda or cpu
            mat_name: name of the MATLAB's file, which contains the pqd signals
            num_of_samples: number of samples in each pqd signal
            fs: sampling frequency of the pqd signal
            mode: training mode - white box, gray box, or black box
            attack_type: the type of the adversarial attack on the model
            best_eps: best value of epsilon with the best results
        """
    main_model = ClassifierDeepCNN()

    if main_model_name == "ResNet18":
        main_model = models.resnet18(weights=None)
        main_model.fc = nn.Linear(in_features=main_model.fc.in_features, out_features=16)

    if mode == 'black box':
        main_model = ClassifierBlackBox()

    test_model = ClassifierDeepCNN()

    if test_model_name == "ResNet18":
        test_model = models.resnet18(weights=None)
        test_model.fc = nn.Linear(in_features=test_model.fc.in_features, out_features=16)

    if test_model_name == "DeepNN_black box":
        test_model = ClassifierBlackBox()

    main_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "models\\" + main_model_name + ".pth"),
                                          map_location=torch.device(device)))
    test_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "models\\" + test_model_name + ".pth"),
                                          map_location=torch.device(device)))

    _, _, cwt_test, labels_test, test_set, _, _, _ = split_train_val_test(cwt_signals_set, labels, seed=0)
    test_loader = DataLoader(test_set, batch_size=16)

    match stage:
        case 'plot':
            pqd_path = os.path.join(os.getcwd(), mat_name + ".mat")
            pqd_set, cwt_set, labels_set = signals_set_maker(pqd_path, cwt_test, labels_test, num_of_samples,
                                                 device, labels_order, train_splitter=0.8)

            if mode == '' and attack_type == '':
                # Base case plotting without any attack:
                plot_pqd_signals('PQD signals', 'PQD signals', pqd_set, labels_order, num_of_samples, fs)
                plot_cwt_signals('CWT signals', 'CWT signals',
                                 test_model, cwt_set, labels_order, le)

                _, _, cm0 = evaluate_classification_performance(test_model, test_loader, device)
                plot_confusion_matrix(cm0, test_model_name, le)

                # Results as function of fixed mode - white/gray/black
                eps_performance_plotter(test_model_name, 'attack', mode='white box')
                eps_performance_plotter(test_model_name, 'attack', mode='gray box')
                eps_performance_plotter(test_model_name, 'attack', mode='black box')

                # Results as function of fixed attack - FGSM/BIM/PGD
                eps_performance_plotter(test_model_name, 'mode', attack_type='FGSM')
                eps_performance_plotter(test_model_name, 'mode', attack_type='BIM')
                eps_performance_plotter(test_model_name, 'mode', attack_type='PGD')

                # Results of retraining process
                eps_performance_plotter('DeepCNN_retrain', 'mode', attack_type='FGSM')
                eps_performance_plotter('DeepCNN_retrain', 'mode', attack_type='BIM')

                # Comparison of the performance results before and after the adversarial training
                eps_performance_plotter('DeepCNN', 'compare', mode='white box', attack_type='FGSM',
                                        retrain_model_name='DeepCNN_retrain')
                eps_performance_plotter('DeepCNN', 'compare', mode='white box', attack_type='BIM',
                                        retrain_model_name='DeepCNN_retrain')

                eps_performance_plotter('DeepCNN', 'compare', mode='gray box', attack_type='FGSM',
                                        retrain_model_name='DeepCNN_retrain')
                eps_performance_plotter('DeepCNN', 'compare', mode='gray box', attack_type='BIM',
                                        retrain_model_name='DeepCNN_retrain')

                eps_performance_plotter('DeepCNN', 'compare', mode='black box', attack_type='FGSM',
                                        retrain_model_name='DeepCNN_retrain')
                eps_performance_plotter('DeepCNN', 'compare', mode='black box', attack_type='BIM',
                                        retrain_model_name='DeepCNN_retrain')

                return

            # Plot CWT signals and confusion matrix for optimal attack parameters, based on its type, mode and model:

            labeled_cwt_set = TensorDataset(cwt_set, labels_set)
            cwt_loader = DataLoader(labeled_cwt_set, batch_size=16)
            adv_set, _ = attacker(main_model, attack_type, cwt_loader, device, epsilon=best_eps, c=1000, kappa=10)
            plot_cwt_signals('CWT signals' + '_' + test_model_name + '_' + mode + '_' + attack_type,
                             'CWT signals attacked by ' + mode + ' ' + attack_type,
                             test_model, adv_set, labels_order, le)
            _, adv_loader = attacker(main_model, attack_type, test_loader, device, epsilon=best_eps)
            _, _, cm = evaluate_classification_performance(test_model, adv_loader, device)
            plot_confusion_matrix(cm, test_model_name, le, attack_type, mode)

        case 'train':
            train_model_name = main_model_name
            train_model = ClassifierDeepCNN()

            if mode == 'black box':
                train_model = ClassifierBlackBox()
                train_model_name = 'DeepNN'

            if main_model_name == "ResNet18":
                train_model = models.resnet18(weights=None)
                train_model.fc = nn.Linear(in_features=train_model.fc.in_features, out_features=16)

            train_pqd_classifier(cwt_signals_set, labels, train_model, train_model_name, device, mode)

        case 'retrain':
            retrain_model = test_model
            mode = 'retrain'
            _, _, _, _, _, _, _, retrain_dataset = split_train_val_test(inputs, labels)
            retrain_loader = DataLoader(retrain_dataset, batch_size=16)

            retrain_inputs = torch.tensor([], dtype=torch.long, device=device)
            retrain_labels = torch.tensor([], dtype=torch.long, device=device)

            steps = 10
            random_start = True
            retrain_epsilons = []

            start = time()
            for i, data in enumerate(retrain_loader):
                batch_inputs, batch_labels = data
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                epsilon = random.uniform(0.015, 0.025)
                retrain_epsilons.append(epsilon)
                alpha = 0.5 * (epsilon / steps)
                attack = torchattacks.PGD(retrain_model, epsilon, alpha, steps, random_start)
                attacked_inputs = attack(batch_inputs, batch_labels)

                retrain_inputs = torch.cat((retrain_inputs, attacked_inputs))
                retrain_labels = torch.cat((retrain_labels, batch_labels))

                batch_time = time()
                print(f"Batch number: {i + 1} of {len(retrain_loader)} | "
                      f"Time elapsed = {(batch_time - start) / 3600:.3f} hours | "
                      f"Epsilon used = {epsilon:.4f}")

            print("Saving retrain set and epsilons used")
            torch.save(retrain_epsilons, os.path.join(os.getcwd(), "retrain_epsilons.pth"))
            torch.save(retrain_inputs, os.path.join(os.getcwd(), "retrain_inputs.pth"))
            torch.save(retrain_labels, os.path.join(os.getcwd(), "retrain_labels.pth"))

            print("Retrain process begins")
            train_pqd_classifier(retrain_inputs, retrain_labels, retrain_model, test_model_name, device, mode)

        case 'evaluate':
            accuracy, f1, _ = evaluate_classification_performance(test_model, test_loader, device)

            print(test_model_name + " training results:\n"f"Accuracy = {accuracy:.2f} | F1 score = {f1:.2f}")
            # plot_confusion_matrix(cm, test_model_name, 2, le)

        case 'attack':
            path_test_accuracy = "accuracies\\" + test_model_name + "_" + mode + "_" + attack_type + \
                                 "_test_accuracy_" + ".pth"
            path_test_mis_class_rate = "mis_class_rates\\" + test_model_name + "_" + mode + "_" + attack_type +  \
                                       "_test_misclassification_rate_" + ".pth"
            path_test_f1_score = "f1_scores\\" + test_model_name + "_" + mode + "_" + attack_type + \
                                 "_test_f1_score_" + ".pth"
            path_test_avg_rob = "average_robustness_scores\\" + test_model_name + "_" + mode + "_" + attack_type +\
                                "_test_average_robustness_" + ".pth"
            path_max_diff_val = "max_diff_values\\" + test_model_name + "_" + mode + "_" + attack_type +\
                                "_test_average_robustness_" + ".pth"

            # for DeepCNN: base accuracy = 0.9, base misclassification rate = 0,
            #                                base f1 score = 0.89, base average_robustness = 0
            # for DeepCNN_retrain: base accuracy = 0.83, base misclassification rate = 0,
            #                                base f1 score = 0.81, base average_robustness = 0

            max_diff_val_vec = [0]
            accuracies = [0.9]
            mis_class_rates = [0]
            f1_scores = [0.89]
            avg_rob_scores = [0]

            if test_model_name == "DeepCNN_retrain":
                accuracies = [0.83]
                f1_scores = [0.81]

            if attack_type == "CW":
                adv_signals, adv_loader = attacker(main_model, attack_type, test_loader, device, c=1000, kappa=10)
                adv_signals_path = test_model_name + "_" + mode + "_" + attack_type + ".pt"
                torch.save(adv_signals, os.path.join(os.getcwd(), adv_signals_path))
                average_robustness_score = calc_avg_robustness(cwt_test, adv_signals)
                mis_class_rate = calc_mcr(test_loader, adv_loader, test_model, device)
                accuracy, f1, _ =\
                    evaluate_classification_performance(test_model, adv_loader, device)

                print(test_model_name + " model, attacked by " + attack_type +
                      " results:\n"f"Accuracy = {accuracy:.2f} | F1 score = {f1:.2f}" +
                      "\nThe biggest difference between the signals is: " +
                      str(torch.max(torch.abs(adv_signals - cwt_test))) + "\nAverage robustness is: " +
                      str(average_robustness_score) + "\nMisclassification rate is: " + str(mis_class_rate))

            else:
                epsilons = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]

                # calculates the model's performance for an adversarial attack, as a function of epsilon
                for i, epsilon in enumerate(epsilons):

                    adv_signals, adv_loader = attacker(main_model, attack_type, test_loader, device, epsilon=epsilon)
                    max_diff_val = torch.max(torch.abs(adv_signals - cwt_test))
                    average_robustness_score = calc_avg_robustness(cwt_test, adv_signals)
                    mis_class_rate = calc_mcr(test_loader, adv_loader, test_model, device)
                    accuracy, f1, _ =\
                        evaluate_classification_performance(test_model, adv_loader, device)

                    max_diff_val_vec.append(max_diff_val)
                    avg_rob_scores.append(average_robustness_score)
                    mis_class_rates.append(mis_class_rate)
                    accuracies.append(accuracy)
                    f1_scores.append(f1)

                    print(test_model_name + " model, attacked by " + attack_type + " with epsilon = " + str(epsilon) +
                          " results:\n"f"Accuracy = {accuracy:.2f} | F1 score = {f1:.2f}" +
                          "\nThe biggest difference between the signals is: " +
                          str(torch.max(torch.abs(adv_signals - cwt_test))) + "\nAverage robustness is: " +
                          str(average_robustness_score) + "\nMisclassification rate is: " + str(mis_class_rate))

                torch.save(max_diff_val_vec, os.path.join(os.getcwd(), path_max_diff_val))
                torch.save(accuracies, os.path.join(os.getcwd(), path_test_accuracy))
                torch.save(mis_class_rates, os.path.join(os.getcwd(), path_test_mis_class_rate))
                torch.save(f1_scores, os.path.join(os.getcwd(), path_test_f1_score))
                torch.save(avg_rob_scores, os.path.join(os.getcwd(), path_test_avg_rob))


if __name__ == '__main__':


    if torch.cuda.is_available():
        device = 'cuda'

    torch.cuda.empty_cache()
    labels_order = ['Normal', 'Sag', 'Swell', 'Interruption', 'Harmonics', 'Oscillatory transient', 'Sag+Harmonics',
                    'Swell+Harmonics', 'Flicker', 'Notch', 'Spike', 'Flicker+Harmonics', 'Interruption+Harmonics',
                    'Flicker+Sag', 'Flicker+Swell', 'Impulsive Transient']
    le = CustomLabelEncoder(labels_order)
    le.fit(np.arange(16))
    # Make and load your dataset
    inputs = torch.load(os.getcwd() + '\\pqd_cwts.pt')
    inputs = torch.movedim(inputs, 3, 1)
    labels = torch.load(os.getcwd() + '\\pqd_labels.pt')

    # Some examples for different stages
    # processing_stage('plot', inputs, labels, labels_order, le, "DeepCNN_gray box", "DeepCNN",
    #                  mode='gray box', attack_type='CW')
    # processing_stage('evaluate', inputs, labels, labels_order, le, "DeepCNN", "ResNet18")
    # processing_stage('train', inputs, labels, labels_order, le,
    #                  "DeepNN_black box", "DeepNN_black box", mode='black box')
    # processing_stage('retrain', inputs, labels, labels_order, le, "DeepCNN", "DeepCNN")
    # processing_stage('attack', inputs, labels, labels_order, le, "DeepNN_black box", "DeepCNN_retrain",
    #                  mode='black box', attack_type='CW')
