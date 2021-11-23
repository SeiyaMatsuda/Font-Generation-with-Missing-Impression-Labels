from trainer.train import pggan_train
from utils.mylib import *
from utils.logger import init_logger
from dataset import *
from models.PGmodel import Generator, Discriminator, StyleDiscriminator
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import FID
from  utils.visualize import visualize_semantic_condition
import pandas as pd

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
    D_model = Discriminator(num_dimension=opts.num_dimension, imp_num=imp_num, char_num=opts.char_num, compress=opts.label_compress).to(opts.device)
    G_model = Generator(weights, latent_size=opts.latent_size, w2v_dimension=w2v_dimension, num_dimension=opts.num_dimension, char_num=opts.char_num, attention=opts.attention, normalize=opts.sc_normalize).to(opts.device)
    G_model_mavg = Generator(weights, latent_size=opts.latent_size, w2v_dimension=w2v_dimension, num_dimension=opts.num_dimension, char_num=opts.char_num, attention=opts.attention, normalize=opts.sc_normalize).to(opts.device)
    if opts.style_discriminator:
        style_D_model = StyleDiscriminator(style_num=4).to(opts.device)
    else:
        style_D_model = None
    fid = FID()
    mAP_score = pd.DataFrame(columns=list(ID.keys()))
    LOGGER.info(f"================Generator================")
    LOGGER.info(f"{G_model}")
    LOGGER.info(f"================Discriminator================")
    LOGGER.info(f"{D_model}")

    #学習済みモデルのパラメータを使用
    # GPUの分散
    if opts.device_count > 1:
        D_model = nn.DataParallel(D_model)
        G_model = nn.DataParallel(G_model)
        G_model_mavg = nn.DataParallel(G_model_mavg)
        if opts.style_discriminator:
            style_D_model = nn.DataParallel(style_D_model)
    # optimizerの定義
    if opts.style_discriminator:
        style_D_optimizer = torch.optim.Adam(style_D_model.parameters(), lr=opts.d_lr, betas=(0, 0.99))
    else:
        style_D_optimizer = None
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr=opts.d_lr, betas=(0, 0.99))
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr=opts.g_lr, betas=(0, 0.99))

    D_TF_loss_list = []
    G_TF_loss_list = []
    D_ch_loss_list = []
    G_ch_loss_list = []
    D_cl_loss_list = []
    G_cl_loss_list = []
    real_acc_list = []
    fake_acc_list = []
    FID_score = []

    transform = Transform()
    #training param
    iter_start = opts.start_iterations
    writer = SummaryWriter(log_dir=opts.learning_log_dir)
    # visualize semantic_condition
    if opts.visualize_sc:
        opts.vsc = visualize_semantic_condition(weights)
        idx = random.sample(list(range(len(data))), 512)
        A = data[idx][:, 0, :, :]
        opts.test_char = torch.eye(opts.char_num)[0].unsqueeze(0).repeat(512, 1)
        opts.test_A = transform(torch.from_numpy(A.astype(np.float32)).clone().unsqueeze(1))
    if opts.multi_learning:
        dataset = Myfont_dataset2(data, label, ID, char_num=opts.char_num,
                              transform=transform)
    else:
        dataset = Myfont_dataset3(data, label, ID, char_num=opts.char_num,
                                  transform=transform)
    bs = opts.batch_size
    for epoch in range(opts.num_epochs):
        start_time = time.time()
        LOGGER.info(f"================epoch_{epoch}================")
        DataLoader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True,
                                                 collate_fn=collate_fn, drop_last=True)
        param = {"opts": opts,  'epoch': epoch, 'G_model': G_model, 'D_model': D_model, 'style_D_model':style_D_model,
                 'G_model_mavg': G_model_mavg, "dataset": dataset, "z": z, "fid": fid, "mAP_score":mAP_score,
                 "Dataset": dataset, 'DataLoader': DataLoader, 'co_matrix': co_matrix,
                 'G_optimizer': G_optimizer, 'D_optimizer': D_optimizer, 'style_D_optimizer': style_D_optimizer, 'log_dir': opts.logs_GAN, "iter_start":iter_start,'ID': ID, 'writer': writer}
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
        FID_score.append(check_point["FID"])
        check_point["mAP_score"].to_csv(os.path.join(opts.learning_log_dir, "impression_AP_score.csv"))
        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        # エポックごとに結果を表示
        LOGGER.info("所要時間 %d 分 %d 秒" % (mins, secs))
        LOGGER.info(f'\tLoss: {check_point["D_epoch_TF_losses"]:.4f}(Discriminator_TF)')
        LOGGER.info(f'\tLoss: {check_point["G_epoch_TF_losses"]:.4f}(Generator_TF)')
        LOGGER.info(f'\tLoss: {check_point["D_epoch_ch_losses"]:.4f}(Discriminator_char)')
        LOGGER.info(f'\tLoss: {check_point["G_epoch_ch_losses"]:.4f}(Generator_char)')
        LOGGER.info(f'\tLoss: {check_point["D_epoch_cl_losses"]:.4f}(Discriminator_class)')
        LOGGER.info(f'\tLoss: {check_point["G_epoch_cl_losses"]:.4f}(Generator_class)')
        LOGGER.info(f'\tacc: {check_point["epoch_real_acc"]:.4f}(real_acc)')
        LOGGER.info(f'\tacc: {check_point["epoch_fake_acc"]:.4f}(fake_acc)')
        LOGGER.info(f'\tFID: {check_point["FID"]:.4f}(FID)')
       # モデル保存のためのcheckpointファイルを作成
        if iter_start >= 100000:
            break

    writer.close()



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
    LOGGER = init_logger(opts.log_dir)
    LOGGER.info(f"================hyper parameter================ \n"
                f"device::{opts.device}\n"
                f"batch_size:{opts.batch_size}\n"
                f"g_lr:{opts.g_lr}\n"
                f"d_lr:{opts.d_lr}\n"
                f"start_iteration:{opts.start_iterations}\n"
                f"res_step:{opts.res_step}\n"
                f"attention:{opts.attention}\n"
                f"multi_learning:{opts.multi_learning}\n"
                f"label_transform:{opts.label_transform}\n"
                f"label_compress:{opts.label_compress}\n"
                f"style_discriminator:{opts.style_discriminator}\n"
                f"img_size:{opts.img_size}\n"
                f"w2v_dimension:{opts.w2v_dimension}\n"
                f"num_dimension:{opts.num_dimension}\n"
                f"latent_size:{opts.latent_size}\n"
                f"num_epochs:{opts.num_epochs}\n"
                f"char_num:{opts.char_num}\n"
                f"num_critic:{opts.num_critic}\n"
                f"lambda_gp:{opts.lambda_gp}\n"
                f"lambda_drift:{opts.lambda_drift}\n"
                f"lambda_style:{opts.lambda_style}\n"
                f"lambda_class:{opts.lambda_class}\n")
    pgmodel_run(opts)
