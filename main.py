from trainer.train import pggan_train
# from models.DCmodel import ACGenerator, ACDiscriminator, CGenerator, CDiscriminator
from utils.mylib import *
from utils.logger import init_logger
from dataset import *
from models.PGmodel import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import FID

def make_logdir(path):
    log_dir = path
    weight_dir = os.path.join(log_dir, 'weight')
    logs_GAN = os.path.join(log_dir, "learning_image")
    learning_log_dir = os.path.join(log_dir, "learning_log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(logs_GAN):
        os.makedirs(logs_GAN)
    if not os.path.exists(learning_log_dir):
        os.makedirs(learning_log_dir)
    return log_dir, weight_dir, logs_GAN, learning_log_dir

def pgmodel_run(opts):
    data = np.array([np.load(d) for d in opts.data])
    label = opts.impression_word_list
    # 生成に必要な乱数
    z_img = torch.randn(4, opts.latent_size * 4 * 4)
    z_cond = torch.randn(4, opts.num_dimension)
    z = (z_img, z_cond)
    #単語IDの変換
    ID = {key:idx+1 for idx, key in enumerate(opts.w2v_vocab)}
    weights = np.array(list(opts.w2v_vocab.values()))
    imp_num = weights.shape[0]
    w2v_dimension = weights.shape[1]
    co_matrix = create_co_matrix(label, ID)
    #モデルを定義
    D_model = Discriminator(num_dimension=opts.num_dimension, imp_num=imp_num, char_num=opts.char_num).to(opts.device)
    G_model = Generator(weights, latent_size=opts.latent_size, w2v_dimension=w2v_dimension, num_dimension=opts.num_dimension, attention=True, char_num=opts.char_num).to(opts.device)
    G_model_mavg = Generator(weights, latent_size=opts.latent_size, w2v_dimension=w2v_dimension, num_dimension=opts.num_dimension, attention=True, char_num=opts.char_num).to(opts.device)
    fid = FID()
    print("Generator:", G_model)
    print("Discriminator:", D_model)
    #学習済みモデルのパラメータを使用
    # GPUの分散
    if opts.device_count > 1:
        D_model = nn.DataParallel(D_model)
        G_model = nn.DataParallel(G_model)
        G_model_mavg = nn.DataParallel(G_model_mavg)

    # optimizerの定義
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr=opts.g_lr, betas=(0, 0.99))
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr=opts.g_lr, betas=(0, 0.99))

    D_TF_loss_list = []
    G_TF_loss_list = []
    D_ch_loss_list = []
    G_ch_loss_list = []
    D_cl_loss_list = []
    G_cl_loss_list = []
    real_acc_list = []
    fake_acc_list = []

    transform = Transform()
    #training param
    iter_start = 0
    writer = SummaryWriter(log_dir=opts.learning_log_dir)
    dataset = Myfont_dataset2(data, label, ID, char_num=opts.char_num,
                              transform=transform)
    bs = opts.batch_size
    pos_weight = dataset.pos_weight

    for epoch in range(opts.num_epochs):
        start_time = time.time()
        LOGGER.info(f"================epoch_{epoch}================")
        DataLoader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True,
                                                 collate_fn=collate_fn, drop_last=True)
        param = {"opts": opts,  'epoch': epoch, 'G_model': G_model, 'D_model': D_model,
                 'G_model_mavg': G_model_mavg, "dataset": dataset, "z": z, "fid": fid,
                 'DataLoader': DataLoader, 'co_matrix': co_matrix, 'pos_weight': pos_weight,
                 'G_optimizer': G_optimizer, 'D_optimizer': D_optimizer, 'log_dir': opts.logs_GAN, "iter_start":iter_start,'ID': ID, 'writer': writer}
        check_point = pggan_train(param)
        iter_start = check_point["iter_finish"]
        D_TF_loss_list.append(check_point["D_epoch_TF_losses"])
        G_TF_loss_list.append(check_point["G_epoch_TF_losses"])
        D_cl_loss_list.append(check_point["D_epoch_cl_losses"])
        G_cl_loss_list.append(check_point["G_epoch_cl_losses"])
        D_ch_loss_list.append(check_point["D_epoch_ch_losses"])
        G_ch_loss_list.append(check_point["G_epoch_ch_losses"])
        real_acc_list.append(check_point["epoch_real_acc"])
        fake_acc_list.append(check_point["epoch_fake_acc"])

        history = {'D_TF_loss': D_TF_loss_list,
         'G_TF_loss': G_TF_loss_list,
         'D_class_loss': D_cl_loss_list,
         'G_class_loss': G_cl_loss_list,
         'D_char_loss': D_ch_loss_list,
         'G_char_loss': G_ch_loss_list
                   }

        accuracy = {'real_acc': real_acc_list,
                    'fake_acc': fake_acc_list}

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        # エポックごとに結果を表示
        LOGGER.info("所要時間 %d 分 %d 秒" % (mins, secs))
        LOGGER.info(f'\tLoss: {check_point["D_epoch_TF_losses"]:.4f}(Discriminator_TF)')
        LOGGER.info(f'\tLoss: {check_point["G_epoch_TF_losses"]:.4f}(Generator_TF)')
        LOGGER.info(f'\tLoss: {check_point["D_epoch_cl_losses"]:.4f}(Discriminator_class)')
        LOGGER.info(f'\tLoss: {check_point["G_epoch_cl_losses"]:.4f}(Generator_class)')
        LOGGER.info(f'\tLoss: {check_point["D_epoch_ch_losses"]:.4f}(Discriminator_char)')
        LOGGER.info(f'\tLoss: {check_point["G_epoch_ch_losses"]:.4f}(Generator_char)')
        LOGGER.info(f'\tacc: {check_point["epoch_real_acc"]:.4f}(real_acc)')
        LOGGER.info(f'\tacc: {check_point["epoch_fake_acc"]:.4f}(fake_acc)')
       # モデル保存のためのcheckpointファイルを作成
        if iter_start >= 100000:
            break

    writer.close()
    return D_TF_loss_list, G_TF_loss_list, D_cl_loss_list, G_cl_loss_list



if __name__=="__main__":
    parser = get_parser()
    opts = parser.parse_args()
    # 再現性確保のためseed値固定
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # make dirs
    opts.log_dir, opts.weight_dir, opts.logs_GAN, opts.learning_log_dir = \
        make_logdir(os.path.join(opts.root, opts.dt_now))
    # 回すモデルの選定
    print(f"device::{opts.device}")
    LOGGER = init_logger(opts.log_dir)
    LOGGER.info(f"================hyper parameter================")
    LOGGER.info(f"batch_size:{opts.batch_size}")
    LOGGER.info(f"g_lr:{opts.g_lr}")
    LOGGER.info(f"d_lr:{opts.d_lr}")
    LOGGER.info(f"img_size:{opts.img_size}")
    LOGGER.info(f"w2v_dimension:{opts.w2v_dimension}")
    LOGGER.info(f"num_dimension:{opts.num_dimension}")
    LOGGER.info(f"latent_size:{opts.latent_size}")
    LOGGER.info(f"num_epochs:{opts.num_epochs}")
    LOGGER.info(f"char_num:{opts.char_num}")

    pgmodel_run(opts)
