"""
Run Bayesian optimization to learn to learn select data for transfer learning.

Uses Python 3.5.
"""

import os
import argparse
import logging
import pickle
import copy

import numpy as np
from scipy import stats
from sklearn.cross_validation import train_test_split
import sys
from robo.fmin import bayesian_optimization
sys.path.insert(0,'emnlp2017_bilstm_cnn_crf')
import task_utils
import data_utils
import similarity
import features
from constants import FEATURE_SETS, SENTIMENT, POS, SLOT_FILLING, POS_BILSTM, PARSING,\
    TASK2TRAIN_EXAMPLES, TASK2DOMAINS, TASKS, POS_PARSING_TRG_DOMAINS, SLOT_FILLING_TRG_DOMAINS, \
    SENTIMENT_TRG_DOMAINS, BASELINES, BAYES_OPT, RANDOM, MOST_SIMILAR_DOMAIN,\
    MOST_SIMILAR_EXAMPLES, ALL_SOURCE_DATA
import random
from bist_parser.bmstparser.src.utils import ConllEntry

def task2_objective_function(task):
    """Returns the objective function of a task."""
    if task == SENTIMENT:
        return objective_function_sentiment
    if task == POS:
        return objective_function_pos
    if task == POS_BILSTM:
        return objective_function_pos_bilstm
    if task == PARSING:
        return objective_function_parsing
    if task == SLOT_FILLING:
        return objective_function_slot_filling
    raise ValueError('No objective function implemented for %s.' % task)


def objective_function_sentiment(feature_weights):
    """
    The objective function to optimize for sentiment analysis.
    :param feature_weights: a numpy array; these are the weights of the features
                            that we want to learn
    :return: the error that should be minimized
    """
    train_subset, train_labels_subset = task_utils.get_data_subsets(
        feature_values, feature_weights, X_train, y_train, SENTIMENT,
        TASK2TRAIN_EXAMPLES[SENTIMENT])

    # train and evaluate the SVM; we input the test documents here but only
    # minimize the validation error
    val_accuracy, _ = task_utils.train_and_evaluate_sentiment(
        train_subset, train_labels_subset, X_val, y_val, X_test, y_test)

    # we minimize the error; the lower the better
    error = 1 - float(val_accuracy)
    return error


def objective_function_pos(feature_weights):
    """
    The objective function to optimize for POS tagging.
    :param feature_weights: a numpy array; these are the weights of the features
                            that we want to learn
    :return: the error that should be minimized
    """
    train_subset, train_labels_subset = task_utils.get_data_subsets(
        feature_values, feature_weights, X_train, y_train, POS,
        TASK2TRAIN_EXAMPLES[POS])

    # train and evaluate the tagger; we input the test documents here but only
    # minimize the validation error
    val_accuracy, _ = task_utils.train_and_evaluate_pos(
        train_subset, train_labels_subset, X_val, y_val)

    # we minimize the error; the lower the better
    error = 1 - float(val_accuracy)
    return error


def objective_function_slot_filling(feature_weights) :

    train_subset, train_labels_subset = task_utils.get_data_subsets(
        feature_values, feature_weights, X_train, y_train, SLOT_FILLING,
        TASK2TRAIN_EXAMPLES[SLOT_FILLING])

    print("Running evaluation for slot filling")
    # train and evaluate the tagger; we input the test documents here but only
    # minimize the validation error
    #print("Training samples {} {}".format(X_train[:10], y_train[:10]))

    #data_utils.dump_to_conll(train_subset, train_labels_subset, "test_dump.txt")


    dev_f1_score, test_f1_score = task_utils.train_and_evaluate_slot_filling_MTL(train_subset, train_labels_subset, X_val, y_val,args=args)

    # we minimize the error; the lower the better
    error = 1 - dev_f1_score

    return error



def objective_function_pos_bilstm(feature_weights):
    """
    The objective function to optimize for POS tagging.
    :param feature_weights: a numpy array; these are the weights of the features
                            that we want to learn
    :return: the error that should be minimized
    """
    train_subset, train_labels_subset = task_utils.get_data_subsets(
        feature_values, feature_weights, X_train, y_train, POS_BILSTM,
        TASK2TRAIN_EXAMPLES[POS_BILSTM])

    # train and evaluate the tagger; we input the test documents here but only
    # minimize the validation error
    val_accuracy, _ = task_utils.train_and_evaluate_pos_bilstm(
        train_subset, train_labels_subset, X_val, y_val)

    # we minimize the error; the lower the better
    error = 1 - float(val_accuracy)
    return error


def objective_function_parsing(feature_weights):
    """
    The objective function to optimize for dependency parsing.
    :param feature_weights: a numpy array; these are the weights of the features
                            that we want to learn
    :return: the error that should be minimized
    """
    train_subset, train_labels_subset = task_utils.get_data_subsets(
        feature_values, feature_weights, X_train, y_train, PARSING,
        TASK2TRAIN_EXAMPLES[PARSING])
    val_accuracy, _ = task_utils.train_and_evaluate_parsing(
        train_subset, train_labels_subset, X_val, y_val,
        parser_output_path=parser_output_path,
        perl_script_path=perl_script_path)
    error = 100 - float(val_accuracy)
    return error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Learn to select data using Bayesian Optimization.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dynet parameters
    parser.add_argument('--dynet-autobatch', type=int,
                        help='use auto-batching (1) (should be first argument)')
    parser.add_argument('--dynet-mem', default=5000, help='the memory used',
                        type=int)  # Note: needs to be given to the script!
    parser.add_argument('--dynet-seed', default=1512141834, type=int,
                        help='the dynet seed')  # Note: needs to still be given!

    # domain and data paths
    parser.add_argument('-d', '--data-path', required=True,
                        help='the path to the directory containing the '
                             'processed_acl or gweb_sancl directory')
    parser.add_argument('-m', '--model-dir', required=True,
                        help='the directory where the model should be saved')
    parser.add_argument('-t', '--trg-domains', nargs='+', required=True,
                        choices=POS_PARSING_TRG_DOMAINS + SENTIMENT_TRG_DOMAINS + SLOT_FILLING_TRG_DOMAINS,
                        help='the domains to which to adapt')
    parser.add_argument('-s', '--src-domains', nargs='+', required=False, choices=SLOT_FILLING_TRG_DOMAINS, default=None,
                        help='the source domains')
    parser.add_argument('--task', choices=TASKS, required=True,
                        help='the task which to optimize')
    parser.add_argument('-b', '--baselines', nargs='+', choices=BASELINES,
                        default=[RANDOM],
                        help='the baselines that should be compared against')
    parser.add_argument('-o', '--parser-output-path',
                        default='outputs', help='the output path of the parser')
    parser.add_argument('-p', '--perl-script-path', help='perl script path',
                        default='bist_parser/bmstparser/src/util_scripts/eval'
                                '.pl')

    # feature parameters
    parser.add_argument('-f', '--feature-sets', nargs='+', default=['similarity'],
                        choices=FEATURE_SETS,
                        help='which feature sets (similarity, topic_similarity,'
                             'word_embedding_similarity, diversity) '
                             'to use; default: similarity')
    parser.add_argument('--z-norm', action='store_true',
                        help='use z-normalisation')  # important to specify
    parser.add_argument('--feature-weights-file',dest="feature_weights_file",
                        help='a file containing learned feature weights to be'
                             'used for cross-domain experiments')

    # word embedding parameters
    parser.add_argument('-wv', '--word2vec-path', help='the path to the word'
                                                       'vector file')
    parser.add_argument('-vs', '--vector-size', type=int, default=300,
                        help='the size of the word vectors')
    parser.add_argument('--header', action='store_true',
                        help='whether the word embeddings file contains header;'
                        'GloVe embeddings used in the paper have no header')

    # processing parameters
    parser.add_argument('-v', '--max-vocab-size', default=10000, type=int,
                        help='the maximum size of the vocabulary')

    # training parameters
    parser.add_argument('--num-iterations', default=10, type=int)
    parser.add_argument('--logging', action='store_true', help='perform logging')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='the number of experiment runs for each domain')
    parser.add_argument('--log-file', required=True,
                        help='the path to which validation and test accuracies'
                             'should be logged')

    # MTL parameters
    parser.add_argument("--mtl-target", dest="mtl_target_task", help="Target Task")
    parser.add_argument("--mtl-aux", dest="mtl_aux_task", nargs='+', help="Auxilliary Task")
    parser.add_argument("--mtl-root-result", dest="mtl_root_dir_result", help="Root directory for results", default="results", type=str)
    parser.add_argument("--mtl-directory-name", dest="mtl_directory_name", help="Directory Name", required=False, type=str)
    parser.add_argument("--mtl-nb-sentence", dest="mtl_nb_sentence", help="Number of training sentence", type=int)
    parser.add_argument("--mtl-batch-range", dest="mtl_batch_range", default=None, type=str)
    parser.add_argument("--mtl-strategy", dest="mtl_strategy", help="Strategy for resource selection", required=False, type=str)
    parser.add_argument("--mtl-ner", dest="mtl_ner", default=0, type=int)
    parser.add_argument("--mtl-ner-name", dest="mtl_ner_name", default=None)
    parser.add_argument("--mtl-epoch",dest="mtl_nb_epoch", help="Number of epoch", default=50, type=int)

    '''
    def construct_param(target_task, root_result, directory_name, strategy=None, nb_sentence=None, ner=0, ner_name=None,
                        diff_level=0, nb_epoch=50, batch_range=None):
    params = construct_param(target_task, "MTL_LTD_Result", "new_result
    '''


    args = parser.parse_args()
    print(sys.path)
    # switch on logging if specified to see the output of LDA training and of
    # the Bayesian optimization
    if args.logging:
        logging.basicConfig(level=logging.INFO)

    assert os.path.exists(args.data_path), ('Error: %s does not exist.' %
                                            args.data_path)
    assert not args.word2vec_path or os.path.exists(args.word2vec_path), \
        'Error: %s does not exist.' % args.word2vec_path

    # create the model directory if it does not exist
    if not os.path.exists(args.model_dir):
        print('Creating %s...' % args.model_dir)
        os.makedirs(args.model_dir)

    # perl script path and parser output path are only required for parsing
    perl_script_path = None

    # get the task-specific methods and hyper-parameters
    num_train_examples = TASK2TRAIN_EXAMPLES[args.task]
    task_trg_domains = TASK2DOMAINS[args.task]

    read_data = data_utils.task2read_data_func(args.task)


    train_and_evaluate = task_utils.task2train_and_evaluate_func(args.task)
    objective_function = task2_objective_function(args.task)

    # get the names of the individual features in the feature sets
    assert args.word2vec_path or 'diversity' not in args.feature_sets,\
        'Error: Word2vec path is required for quadratic entropy in ' \
        'diversity-based features.'
    feature_names = features.get_feature_names(args.feature_sets)

    if args.feature_weights_file:
        print('Training model with pre-learned feature weights rather than '
              'learning new ones...')
        assert os.path.exists(args.feature_weights_file),\
            'Error: %s does not exist.' % args.feature_weights_file

    # read the data and pickle it or load it
    preproc_data_path = os.path.join(args.model_dir,
                                     'preproc_data_%s.pkl' % args.task)

    domain2data = read_data(args.data_path)
    print('Saving domain2data object to %s...' % preproc_data_path)
    with open(preproc_data_path, 'wb') as f:
        pickle.dump(domain2data, f)

    assert set(task_trg_domains) == set(domain2data.keys())

    # create the vocabulary or load it if it was already created


    vocab_path = os.path.join(args.model_dir, 'vocab.txt')
    vocab = data_utils.Vocab(args.max_vocab_size, vocab_path)
    # retrieve all available tokenised sentences
    tokenised_sentences = data_utils.get_all_docs(domain2data.items(), unlabeled=False)[0]
    vocab.create(tokenised_sentences)
    del tokenised_sentences


    # load word vectors if we are using them
    word2vec = None
    if args.word2vec_path:
        vocab_word2vec_file = os.path.join(args.model_dir, 'vocab_word2vec.txt')
        word2vec = similarity.load_word_vectors(
            args.word2vec_path, vocab_word2vec_file, vocab.word2id,
            vector_size=args.vector_size, header=args.header)

    if args.task in [POS, POS_BILSTM, SLOT_FILLING]:
        print('Using words as training data for POS tagging, slot filling...')
        domain2train_data = domain2data
    else:
        raise ValueError('Data preproc for %s is not implemented.' % args.task)

    print('Creating relative term frequency distributions for all domains...')
    term_dist_path = os.path.join(args.model_dir, 'term_dist.txt')
    domain2term_dist = similarity.get_domain_term_dists(
        term_dist_path, domain2data, vocab)

    import time



    # perform optimization for every target domain
    for trg_domain in args.trg_domains:
        print('Target domain:', trg_domain)

        # get the training data of all source domains (not the target domain)
        if args.src_domains is not None :

            X_train, y_train, train_domains = data_utils.get_all_docs(
                [(k, v) for (k, v) in sorted(domain2train_data.items())
                 if k != trg_domain and k in args.src_domains], unlabeled=False)

            # get the unprocessed examples for extracting the feature values
            examples, y_train_check, train_domains_check = data_utils.get_all_docs(
                [(k, v) for (k, v) in sorted(domain2data.items())
                 if k != trg_domain and k in args.src_domains], unlabeled=False)

            print("Reading source domain training data, total : {}".format(len(examples)))
            print("Type X_train : {} y_train : {} train_domains : {}".format(type(X_train), type(y_train), type(train_domains)))
            #print("Domains : {}".format(train_domains))
        else :
            # Ruder's implementation
            X_train, y_train, train_domains = data_utils.get_all_docs(
                [(k, v) for (k, v) in sorted(domain2train_data.items())
                 if k != trg_domain], unlabeled=False)

            # get the unprocessed examples for extracting the feature values
            examples, y_train_check, train_domains_check = data_utils.get_all_docs(
                [(k, v) for (k, v) in sorted(domain2data.items())
                 if k != trg_domain], unlabeled=False)

        # some sanity checks just to make sure the processed and the
        # unprocessed data still correspond to the same examples
        assert np.array_equal(y_train, y_train_check)
        assert len(train_domains) == len(train_domains_check),\
            'Error: %d != %d.' % (len(train_domains), len(train_domains_check))
        assert train_domains == train_domains_check, ('Error: %s != %s' % (
            str(train_domains), str(train_domains_check)))
        if args.task in [POS, POS_BILSTM, PARSING, SLOT_FILLING]:
            # for sentiment, we are using a sparse matrix
            X_train = np.array(X_train)

        print('Training data shape:', X_train.shape, y_train.shape)

        # train topic model if any of the features requires a topic distribution
        topic_vectorizer, lda_model = None, None
        if any(f_name.startswith('topic') for f_name in feature_names):
            # train a topic model on labeled and unlabeled data of all domains
            topic_vectorizer, lda_model = similarity.train_topic_model(
                data_utils.get_all_docs(
                    domain2data.items(), unlabeled=True)[0], vocab)

        # get the feature representations of the training data
        print('Creating the feature representations for the training data. '
              'This may take some time...')
        start_time = time.time()
        feature_values = features.get_feature_representations(
            feature_names, examples, domain2data[trg_domain][0], vocab,
            word2vec, topic_vectorizer, lda_model)

        if args.z_norm:
            # apply z-normalisation; this is important for good performance
            print('Z-normalizing features...')
            print('First five example features before normalisation:',
                  feature_values[:5, :])
            print('Standard deviation of features:', np.std(feature_values,
                                                            axis=0))
            print('Mean of features:', np.mean(feature_values, axis=0))
            feature_values = stats.zscore(feature_values, axis=0)

        # delete unnecessary variables to save space
        del examples, y_train_check, train_domains_check

        # run num_runs iterations of the optimization and baselines in order to
        # compute statistics around mean/variance; things that vary between
        # runs: validation/test split; train set of random baseline;
        # final BayesOpt parameters; the feature values are constant for each
        # run, which is why we generate them before to reduce the overhead
        run_dict = {method: [] for method in BASELINES + [BAYES_OPT]}

        for i in range(args.num_runs):
            print('\nTarget domain %s. Run %d/%d.' % (trg_domain, i+1,
                                                      args.num_runs))

            # get the evaluation data from the target domain
            # X_test, y_test, _ = domain2train_data[trg_domain]
            X_test, y_test, _  = read_data(args.data_path, fold=['test'], domain_=[trg_domain])[trg_domain]
            X_val, y_val, _ = read_data(args.data_path, fold=['dev'], domain_=[trg_domain])[trg_domain]

            # split off a validation set from the evaluation data
            # X_test, X_val, y_test, y_val = train_test_split(
            #     X_test, y_test, test_size=100, stratify=y_test
            #     if args.task == SENTIMENT else None)
            print('# of validation examples: %d. # of test examples: %d.'
                  % (len(y_val), len(y_test)))


            # train the model with pre-learned feature weights if specified
            if args.feature_weights_file:
                print('Training with pre-learned feature weights...')
                task_utils.train_pretrained_weights(
                    feature_values, X_train, y_train, train_domains,
                    num_train_examples, X_val, y_val, X_test, y_test,
                    trg_domain, args, feature_names, parser_output_path=None,
                    perl_script_path=None)
                exit(2)
            for baseline in args.baselines:
                break
                # select the training data dependent on the baseline
                if baseline == RANDOM:
                    print('Randomly selecting examples...')
                    train_subset, _, labels_subset, _ = train_test_split(
                        X_train, y_train, train_size=num_train_examples,
                        stratify=y_train if args.task == SENTIMENT else None)
                elif baseline == ALL_SOURCE_DATA:
                    print('Selecting all source data examples...')
                    train_subset, labels_subset = X_train, y_train
                elif baseline == MOST_SIMILAR_DOMAIN:
                    print('Selecting examples from the most similar domain...')
                    most_similar_domain = similarity.get_most_similar_domain(
                        trg_domain, domain2term_dist)
                    train_subset, labels_subset, _ = domain2train_data[
                        most_similar_domain]
                    train_subset, _, labels_subset, _ = train_test_split(
                        train_subset, labels_subset, train_size=num_train_examples,
                        stratify=labels_subset if args.task == SENTIMENT else None)
                elif baseline == MOST_SIMILAR_EXAMPLES:
                    print('Selecting the most similar examples...')
                    one_all_weights = np.ones(len(feature_names))
                    one_all_weights[1:] = 0
                    train_subset, labels_subset = task_utils.get_data_subsets(
                        feature_values, one_all_weights, X_train, y_train,
                        args.task, num_train_examples)
                else:
                    raise ValueError('%s is not a baseline.' % baseline)

                # train the baseline
                val_accuracy, test_accuracy = train_and_evaluate(
                    train_subset, labels_subset, X_val, y_val,
                    X_test, y_test, parser_output_path=parser_output_path,
                    perl_script_path=perl_script_path)
                run_dict[baseline].append((val_accuracy, test_accuracy))

            # define the lower and upper bounds of the input space [-1, 1]
            lower = np.array(len(feature_names) * [-1])
            upper = np.array(len(feature_names) * [1])
            print('Lower limits shape:', lower.shape)
            print('Upper limits shape:', upper.shape)

            print('Running Bayesian Optimization...')
            res = bayesian_optimization(objective_function, lower=lower,
                                        upper=upper,
                                        num_iterations=args.num_iterations)

            best_feature_weights = res['x_opt']
            print('Best feature weights', best_feature_weights)

            print("Selecting data based on best feature weights :")
            train_subset, labels_subset = task_utils.get_data_subsets(
                feature_values, best_feature_weights, X_train, y_train,
                args.task, num_train_examples)
            val_accuracy, test_accuracy = train_and_evaluate(
                train_subset, labels_subset, X_val, y_val, X_test, y_test, args=args, BO=False)
            run_dict[BAYES_OPT].append((val_accuracy, test_accuracy,
                                          best_feature_weights))

            # your script
            elapsed_time = time.time() - start_time
            print("ELAPSED TIME : {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
        # log the results of all methods to the log file
        data_utils.log_to_file(args.log_file, run_dict, trg_domain, args)
