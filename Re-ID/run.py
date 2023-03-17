from embed import embed, evaluate
from train import train

def run(name, load_path, store_path, load_path_query, store_path_query, num_it = 30000):
    net_path = 'res/' + name + '_mainNet.pkl'
    head_path = 'res/' + name + '_mainHead.pkl'

    loss_test, loss_train, epochs = train(name, num_it)
    print(loss_train)
    print(loss_test)
    print(epochs)

    embed(load_path,store_path, net_path=net_path, head_path=head_path)
    embed(load_path_query, store_path_query, net_path=net_path, head_path=head_path)
    evaluate('res/embd_res', 'res/embd_query')

if __name__ == '__main__':
    load_path = '/mnt/analyticsvideo/DensePoseData/market1501/bounding_box_test'
    store_path = 'res/embd_res'
    load_path_query = '/mnt/analyticsvideo/DensePoseData/market1501/query'
    store_path_query = 'res/embd_query'
    name = 'testing'
    run(name, load_path, store_path, load_path_query, store_path_query, num_it = 40)
    