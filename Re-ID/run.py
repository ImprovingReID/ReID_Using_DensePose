import logging
import os
from pathlib import Path
from embed import embed, evaluate
from train import train
import matplotlib.pyplot as plt
import pickle

def run(name, load_path, store_path, load_path_query, store_path_query, num_it = 30000):
    path_log = Path("log/" + name + "train.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    try:
        if path_log.exists():
            raise FileExistsError
    except FileExistsError:
        print("Already exist log file: {}".format(path_log))
        print("\nDo you really want to delete and overwrite \ {} \" ? (y/n) ", end="".format(path_log))
        ch = input()
        if ch=='y':
            try:
                os.remove(path_log)
                print("\nThe File, \ {} \" deleted successfully!".format(path_log))
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                    datefmt="%a, %d %b %Y %H:%M:%S",
                    filename=path_log.__str__(),
                    filemode="w",
                )
            except IOError:
                print("\nThe file \ {} \" is not available in the directory!".format(path_log))
        else:
            print("\nExiting...")
            raise
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=path_log.__str__(),
            filemode="w",
        )
        print("Create log file: {}".format(path_log))
        
    net_path = 'res/' + name + '_mainNet.pkl'
    head_path = 'res/' + name + '_mainHead.pkl'

    loss_test, loss_train, epochs = train(name, load_path, num_it)
    print(loss_train)
    print(loss_test)
    print(epochs)

    with open('res/' + name + '_loss_train', "wb") as fp:   #Pickling
        pickle.dump(loss_train, fp)
    with open('res/' + name + '_loss_test', "wb") as fp:   #Pickling
        pickle.dump(loss_test, fp)
    with open('res/' + name + '_loss_epochs', "wb") as fp:   #Pickling
        pickle.dump(epochs, fp)

    embed(load_path + 'bounding_box_test', store_path, net_path=net_path, head_path=head_path)
    embed(load_path_query, store_path_query, net_path=net_path, head_path=head_path)
    evaluate(store_path, store_path_query)

def loss_plot(name):
    pass

if __name__ == '__main__':
    name = 'testing'
    load_path = '/mnt/analyticsvideo/DensePoseData/market1501/'
    store_path = 'res/' + name + 'embd_res'
    load_path_query = '/mnt/analyticsvideo/DensePoseData/market1501/query'
    store_path_query = 'res/' + name + 'embd_query'
    run(name, load_path, store_path, load_path_query, store_path_query, num_it = 30000)
    