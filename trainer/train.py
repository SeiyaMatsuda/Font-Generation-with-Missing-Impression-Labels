from utils.mylib import *
from utils.loss import *
from utils.visualize import *
from dataset import *
def gradient_penalty(netD, real, fake, cond, res, batch_size, gamma=1):
    device = real.device
    alpha = torch.rand(batch_size, 1, 1, 1, requires_grad=True).to(device)
    x = alpha*real + (1-alpha)*fake
    d_= netD.forward(x, res, cond=cond)[0]
    g = torch.autograd.grad(outputs=d_, inputs=x,
                            grad_outputs=torch.ones(d_.shape).to(device),
                            create_graph=True, retain_graph=True,only_inputs=True)[0]
    g = g.reshape(batch_size, -1)
    return ((g.norm(2,dim=1)/gamma-1.0)**2).mean()

def pggan_train(param):
    # paramの変数
    epoch = param['epoch']
    opts = param["opts"]
    G_model = param["G_model"]
    D_model = param["D_model"]
    fid = param['fid']
    DataLoader = param["DataLoader"]
    # label_weight = param['label_weight']
    co_matrix = param['co_matrix']
    pos_weight = param['pos_weight']
    ID = param['ID']
    test_z = param["z"]
    G_optimizer = param["G_optimizer"]
    D_optimizer = param["D_optimizer"]
    iter_start = param["iter_start"]
    G_model_mavg = param["G_model_mavg"]
    writer = param['writer']
    ##training start
    G_model.train()
    D_model.train()
    iter = iter_start
    if iter == opts.res_step * 5.5:
        G_optimizer.param_groups[0]['lr'] = opts.g_lr/5
        D_optimizer.param_groups[0]['lr'] = opts.d_lr/5
    #lossの初期化
    D_running_TF_loss = 0
    G_running_TF_loss = 0
    D_running_cl_loss = 0
    G_running_cl_loss = 0
    G_running_char_loss = 0
    D_running_char_loss = 0
    real_acc = []
    fake_acc = []
    fid_score = []
    #Dataloaderの定義
    databar = tqdm.tqdm(DataLoader)
    #マルチクラス分類
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(opts.device)
    kl_loss = KlLoss(activation='softmax').to(opts.device)
    ca_loss = CALoss()
    mse_loss = torch.nn.SmoothL1Loss().to(opts.device)
    for batch_idx, samples in enumerate(databar):
        real_img, char_class, labels = samples['img_target'], samples['charclass_target'], samples['multi_embed_label_target']
        # real_img, char_class, labels = samples['img'], samples['charclass'], samples['embed_label']
        #ステップの定義
        res = iter / opts.res_step
        # get integer by floor
        #image size define
        eps = 1e-7
        n = min(res, len(G_model.module.blocks))
        n_layer = max(int(n - eps), 0)
        img_size = G_model.module.size[n_layer]
        real_img = F.adaptive_avg_pool2d(real_img, (img_size, img_size))

        # バッチの長さを定義
        batch_len = real_img.size(0)
        #デバイスの移
        real_img,  char_class = \
            real_img.to(opts.device), char_class.to(opts.device)
        # 文字クラスのone-hotベクトル化
        char_class_oh = torch.eye(opts.char_num)[char_class].to(opts.device)
        # 印象語のベクトル化
        labels_oh = Multilabel_OneHot(labels, len(ID), normalize=True)
        missing_prob = missing2prob(labels_oh, co_matrix).to(opts.device)
        # labels_oh = missing2clean(missing_prob).to(opts.device)
        # labels_oh = torch.eye(len(ID))[labels-1].to(opts.device)
        # training Generator
        #画像の生成に必要なノイズ作成
        z_img = torch.randn(batch_len, opts.latent_size * 16)
        z_cond = torch.randn(batch_len, opts.num_dimension)
        z = (z_img, z_cond)
        ##画像の生成に必要な印象語ラベルを取得
        _, _, D_real_class = D_model(real_img, res)
        gen_label_ = F.softmax(D_real_class.detach(), dim=1)
        # gen_label_ = torch.sigmoid(D_real_class.detach())
        gen_label = (gen_label_ - gen_label_.mean(0)) / (gen_label_.std(0) + 1e-7)
        # ２つのノイズの結合
        fake_img, mu, logvar = G_model(z, char_class_oh, gen_label, res)
        D_fake_TF, D_fake_char, D_fake_class = D_model(fake_img, res, cond=mu)
        # Wasserstein lossの計算
        G_TF_loss = -torch.mean(D_fake_TF)
        # 文字クラス分類のロス
        G_char_loss = kl_loss(D_fake_char, char_class_oh)
        # 印象語分類のロス
        G_class_loss = kl_loss(D_fake_class, gen_label_)
        # G_class_loss = mse_loss(torch.sigmoid(D_fake_class), gen_label_) * 1000
        G_kl_loss = ca_loss(mu, logvar)
        # mode seeking lossの算出

        G_loss = G_TF_loss + G_char_loss + G_class_loss + G_kl_loss
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        G_running_TF_loss += G_TF_loss.item()
        G_running_cl_loss += G_class_loss.item()
        G_running_char_loss += G_char_loss.item()

        # update netG_mavg by moving average
        momentum = 0.995  # remain momentum
        alpha = min(1.0 - (1 / (iter + 1)), momentum)
        for p_mavg, p in zip(G_model_mavg.parameters(), G_model.parameters()):
            p_mavg.data = alpha * p_mavg.data + (1.0 - alpha) * p.data

        #training Discriminator
        #Discriminatorに本物画像を入れて順伝播⇒Loss計算
        for _ in range(opts.num_critic):
            # 生成用のラベル
            fake_img, mu, _ = G_model(z, char_class_oh, gen_label, res)
            D_real_TF, D_real_char, D_real_class = D_model(real_img, res, cond=mu)
            D_real_loss = -torch.mean(D_real_TF)
            D_fake, D_fake_char, _ = D_model(fake_img.detach(), res, cond=mu)
            D_fake_loss = torch.mean(D_fake)
            gp_loss = gradient_penalty(D_model, real_img.data, fake_img.data, mu, res, real_img.shape[0])
            loss_drift = (D_real_TF ** 2).mean()

            #Wasserstein lossの計算
            D_TF_loss = D_fake_loss + D_real_loss + opts.lambda_gp * gp_loss
            # 文字クラス分類のロス
            D_char_loss = (kl_loss(D_real_char, char_class_oh) + kl_loss(D_fake_char, char_class_oh)) / 2
        # 印象語分類のロス
            D_class_loss = kl_loss(D_real_class, missing_prob)
        #     D_class_loss = bce_loss(D_real_class, missing_prob)
            D_loss = D_TF_loss + D_char_loss + D_class_loss + loss_drift * 0.001
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()
            D_running_TF_loss += D_TF_loss.item()
            D_running_cl_loss += D_class_loss.item()
            D_running_char_loss += D_char_loss.item()
        ##caliculate accuracy
        real_pred = 1 * (torch.sigmoid(D_real_TF) > 0.5).detach().cpu()
        fake_pred = 1 * (torch.sigmoid(D_fake) > 0.5).detach().cpu()
        real_TF = torch.ones(real_pred.size(0))
        fake_TF = torch.zeros(fake_pred.size(0))
        r_acc = (real_pred == real_TF).float().sum().item() / len(real_pred)
        f_acc = (fake_pred == fake_TF).float().sum().item()/len(fake_pred)
        real_acc.append(r_acc)
        fake_acc.append(f_acc)



        ##tensor bord
        writer.add_scalars("TF_loss", {'D_TF_loss': D_TF_loss, 'G_TF_loss': G_TF_loss}, iter)
        writer.add_scalars("class_loss", {'D_class_loss': D_class_loss, 'G_class_loss': G_class_loss}, iter)
        writer.add_scalars("char_loss", {'D_char_loss': D_char_loss, 'G_char_loss': G_char_loss}, iter)
        writer.add_scalars("Acc", {'real_acc': r_acc, 'fake_acc': f_acc}, iter)

        iter += 1

        if iter % 100 == 0:
            test_label = ['decorative', 'big', 'shade', 'manuscript', 'ghost']
            test_emb_label = [[ID[key]] for key in test_label]
            label = Multilabel_OneHot(test_emb_label, len(ID), normalize=False)
            save_path = os.path.join(opts.logs_GAN, 'img_iter_%05d_%02d✕%02d.png' % (iter, real_img.size(2), real_img.size(3)))
            visualizer(save_path, G_model_mavg, test_z, opts.char_num, label, res, opts.device)
            G_model_mavg.train()
        if iter % 1000 == 0:
            weight = {'G_net': G_model_mavg.state_dict(),
                   'G_optimizer': G_optimizer.state_dict(),
                   'D_net': D_model.state_dict(),
                   'D_optimizer': D_optimizer.state_dict()}
            torch.save(weight, os.path.join(opts.weight_dir, 'weight_iter_%d.pth' % (iter)))
        if iter==100000:
            break

    fid_disttance = fid.calculate_fretchet(real_img.data.cpu().repeat(1, 3, 1, 1),
                                           fake_img.data.cpu().repeat(1, 3, 1, 1),  cuda=opts.device, batch_size=opts.batch_size//4)
    writer.add_scalar("fid", fid_disttance, epoch)
    D_running_TF_loss /= len(DataLoader)
    G_running_TF_loss /= len(DataLoader)
    D_running_cl_loss /= len(DataLoader)
    G_running_cl_loss /= len(DataLoader)
    D_running_char_loss /= len(DataLoader)
    G_running_char_loss /= len(DataLoader)
    real_acc = sum(real_acc)/len(real_acc)
    fake_acc = sum(fake_acc)/len(fake_acc)
    check_point = {
                   'D_epoch_TF_losses': D_running_TF_loss,
                   'G_epoch_TF_losses': G_running_TF_loss,
                   'D_epoch_cl_losses': D_running_cl_loss,
                   'G_epoch_cl_losses': G_running_cl_loss,
                   'D_epoch_ch_losses': D_running_char_loss,
                   'G_epoch_ch_losses': G_running_char_loss,
                   'epoch_real_acc': real_acc,
                  'epoch_fake_acc':fake_acc,
                   "iter_finish": iter,
                   "res": res
                   }
    return check_point