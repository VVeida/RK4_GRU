import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Uitls.Function import safemakefile,setup_seed,one_step_track_RK4GRU,split_gm_GRU,load_matdata,redirect_stdout,ScheduledSampler
from Uitls.Network import myDNN,myRK4GRUcell


def train_RK4PIGRU_main(args):
    modelsave_path = args.modelsave_path
    safemakefile(modelsave_path)
    ###################################################################################################
    train_id = [7]
    train_random_indices,data_all = load_matdata(args, 'train',train_id)
    data_all = np.transpose(data_all, (0, 2, 1))
    absxmax = np.max(np.abs(data_all[0,0,:]))
    data_input, data_output = one_step_track_RK4GRU(data_all, args)
    indices = np.arange(data_input.shape[0])
    np.random.shuffle(indices)
    train_indices, val_indices = train_test_split(indices, test_size=0.2)
    train_input_gm = data_input[train_indices, :, :]
    train_output_gm = data_output[train_indices, :, :]
    val_input_gm = data_input[val_indices, :, :]
    val_output_gm = data_output[val_indices, :, :]
    train_input, train_output, train_gm = split_gm_GRU(train_input_gm,train_output_gm)
    val_input, val_output, val_gm = split_gm_GRU(val_input_gm,val_output_gm)
    train_dataset = torch.utils.data.TensorDataset(train_input, train_output,train_gm)
    val_dataset = torch.utils.data.TensorDataset(val_input, val_output, val_gm )
    ######################################################################################################
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
    RK4GRUcell = myRK4GRUcell(args).to(args.device)
    scheduler = ScheduledSampler(args.epochs, args.stage1, args.stage2)
    optimizer = torch.optim.Adam( RK4GRUcell.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.lr_step, args.lr_gamma)
    criterion = torch.nn.MSELoss(reduction='none')
    train_epochs_loss = []
    val_epochs_loss = []
    best_loss = torch.tensor(float('inf'))
    best_epoch = 0
    for epoch in range(args.epochs):
        RK4GRUcell.train()
        train_epoch_loss = []
        tf_ratio, stage = scheduler.teacher_forcing_ratio(epoch)
        # ===============================================
        for idx, (SVi, SVjtarget,exc) in enumerate(train_dataloader):
            use_tf = torch.rand(1) <= tf_ratio  # teacher_forcing
            SVi = SVi.to(torch.float32).to(args.device)
            SVjtarget = SVjtarget.to(torch.float32).to(args.device)
            SVj = torch.zeros((SVjtarget.shape[0],SVjtarget.shape[1],SVjtarget.shape[2])).to(torch.float32).to(args.device)
            rj = torch.zeros((SVjtarget.shape[0], SVjtarget.shape[1], args.SV_feature)).to(torch.float32).to(args.device)
            zj = torch.zeros((SVjtarget.shape[0], SVjtarget.shape[1], args.SV_feature)).to(torch.float32).to(args.device)
            for gru_s in range(args.gru_step):
                exci = exc[:,gru_s:gru_s+1,:].to(torch.float32).to(args.device)
                excj = exc[:,gru_s+1:gru_s+2,:].to(torch.float32).to(args.device)
                if use_tf:
                    if gru_s == 0:
                        SV_next,r,z = RK4GRUcell(SVi,exci,excj)
                        SVj[:,gru_s:gru_s+1,:] = SV_next
                        rj[:,gru_s:gru_s+1,:] = r
                        zj[:,gru_s:gru_s+1,:] = z
                    else:
                        SV_next,r,z = RK4GRUcell(SVjtarget[:,gru_s-1:gru_s,:],exci,excj)
                        SVj[:,gru_s:gru_s+1,:] = SV_next
                        rj[:,gru_s:gru_s+1,:] = r
                        zj[:,gru_s:gru_s+1,:] = z
                else:
                    if gru_s == 0:
                        SV_next,r,z = RK4GRUcell(SVi,exci,excj)
                        SVj[:,gru_s:gru_s+1,:] = SV_next
                        rj[:,gru_s:gru_s+1,:] = r
                        zj[:,gru_s:gru_s+1,:] = z
                    else:
                        SV_next,r,z = RK4GRUcell(SV_next,exci,excj)
                        SVj[:,gru_s:gru_s+1,:] = SV_next
                        rj[:,gru_s:gru_s+1,:] = r
                        zj[:,gru_s:gru_s+1,:] = z
            # loss_pred = criterion(SVj, SVjtarget)
            # loss
            weight = torch.ones_like(SVjtarget)
            loss_mse = criterion(SVj, SVjtarget)
            weight[:,:,0] = weight[:,:,0] + torch.abs(SVjtarget[:,:,0])/absxmax
            weight = torch.square(weight)
            loss_pred = torch.mean( loss_mse*weight)
            if stage == 1:
                loss_ru = (torch.nn.MSELoss()(rj, torch.ones_like(rj).to(torch.float32).to(args.device)) +
                           torch.nn.MSELoss()(zj, torch.ones_like(zj).to(torch.float32).to(args.device)))
                loss = loss_pred + loss_ru
            else:
                loss = loss_pred
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(RK4GRUcell.parameters(), 2.0)
            optimizer.step()
            train_epoch_loss.append(loss_pred.cpu().detach().numpy())
        train_epochs_loss.append([epoch, np.average(train_epoch_loss)])
        with redirect_stdout():
            print('###################### epoch_{} ######################'.format(epoch))
            print("[train  lr={}]".format( lr_scheduler.get_last_lr()[0]))
            print("loss={}".format(np.average(train_epoch_loss)))
        lr_scheduler.step()
         # ===============================================
        if epoch % args.valper == 0 or epoch==args.epochs-1:
            RK4GRUcell.eval()
            val_epoch_loss = []
            for idx, (SVi, SVjtarget,exc) in enumerate(val_dataloader):
                SVi = SVi.to(torch.float32).to(args.device)
                SVjtarget = SVjtarget.to(torch.float32).to(args.device)
                SVj = torch.zeros((SVjtarget.shape[0], SVjtarget.shape[1], SVjtarget.shape[2])).to(torch.float32).to(args.device)
                rj = torch.zeros((SVjtarget.shape[0], SVjtarget.shape[1], args.SV_feature)).to(torch.float32).to(args.device)
                zj = torch.zeros((SVjtarget.shape[0], SVjtarget.shape[1], args.SV_feature)).to(torch.float32).to(args.device)
                for gru_s in range(args.gru_step):
                    exci = exc[:,gru_s:gru_s+1,:].to(torch.float32).to(args.device)
                    excj = exc[:,gru_s+1:gru_s+2,:].to(torch.float32).to(args.device)
                    if gru_s == 0:
                        SV_next,r,z = RK4GRUcell(SVi,exci,excj)
                        SVj[:, gru_s:gru_s + 1, :] = SV_next
                        rj[:, gru_s:gru_s + 1, :] = r
                        zj[:, gru_s:gru_s + 1, :] = z
                    else:
                        SV_next,r,z = RK4GRUcell(SV_next,exci,excj)
                        SVj[:, gru_s:gru_s + 1, :] = SV_next
                        rj[:, gru_s:gru_s + 1, :] = r
                        zj[:, gru_s:gru_s + 1, :] = z

                # loss
                weight = torch.ones_like(SVjtarget)
                loss_mse = criterion(SVj, SVjtarget)
                weight[:, :, 0] = weight[:, :, 0] + torch.abs(SVjtarget[:, :, 0])/absxmax
                weight = torch.square(weight)
                loss_pred = torch.mean(loss_mse * weight)
                loss = loss_pred
                val_epoch_loss.append(loss.cpu().detach().numpy() )
            val_epochs_loss.append([epoch, np.average(val_epoch_loss)])
            with redirect_stdout():
                print("[val]  loss={}".format(np.average(val_epoch_loss)), end='\n')
            if (np.average(val_epoch_loss) < best_loss) and stage == 3:
                best_loss = np.average(val_epoch_loss)
                best_epoch = epoch
                torch.save(RK4GRUcell.state_dict(), modelsave_path + 'RK4GRUcell_best.pth')
    ##########################################################################################################
    train_epochs_loss = np.array(train_epochs_loss)
    val_epochs_loss = np.array(val_epochs_loss)
    np.save( modelsave_path +'train_epochs_loss.npy', train_epochs_loss)
    np.save( modelsave_path +'val_epochs_loss.npy', val_epochs_loss)
    return train_random_indices,train_epochs_loss,val_epochs_loss,best_epoch

def test_RK4PIGRU_main(args,runmodel = None,test_random_indices = None):
    modelsave_path = args.modelsave_path
    #####################################################################################################
    random_indices,data_all = load_matdata(args, runmodel,test_random_indices)
    with open(args.modelsave_path + "run.txt", "a") as file:
        file.write(f"\n test IDs: {random_indices}")
    # state
    test_x_xdot = data_all[:,:, :2] # (1,1499,2)
    test_g = - data_all[:, :, 2] - data_all[:, :, 3]
    test_state = np.concatenate((test_x_xdot, test_g[:, :, np.newaxis]), axis=2)   # (20,1499,3)  # [x xdot g]
    # gm
    test_exc = data_all[:,:,2:3]   # (1,1499,1)
    exc = torch.from_numpy(test_exc)
    ##########################################################################################################
    RK4GRUcell = myRK4GRUcell(args).to(args.device)
    model_path = modelsave_path + 'RK4GRUcell_best.pth'
    RK4GRUcell.load_state_dict(torch.load(model_path))
    RK4GRUcell.eval()
    ######################################################################################################
    pred_state = np.zeros_like(test_state)  # (1,1499,3) # [x xdot g]
    pred_state[:, 0, :] = test_state[:, 0, :]
    for i in tqdm(range(pred_state.shape[1] - 1), desc='Predict tracks'):
        svi = pred_state[:,i:i+1, :]
        svi = torch.from_numpy(svi)
        svi = svi.to(torch.float32).to(args.device)
        exci = exc[:,i:i + 1, :].to(torch.float32).to(args.device)
        excj = exc[:,i+1:i+2, :].to(torch.float32).to(args.device)
        svj, r, z = RK4GRUcell(svi, exci, excj)
        svj = svj.cpu().detach().numpy()
        pred_state[:,i+1:i+2, :] = svj
    rho_array = np.zeros([pred_state.shape[0],pred_state.shape[-1]])
    for test_id in range(pred_state.shape[0]):
        for j in range(pred_state.shape[-1]):   #(20,1499,3)  # [x xdot g]
            rhoj = np.corrcoef(pred_state[test_id,:, j], test_state[test_id,:, j])[1, 0]
            rho_array[test_id,j] = rhoj
            with open(args.modelsave_path + "run.txt", "a") as file:
                file.write(f"\n TEST_Num.{test_id+1} Response{j+1} CC: {rhoj:.5f}")
        with open(args.modelsave_path + "run.txt", "a") as file:
            file.write(f"\n")
    rho = np.average(rho_array[:,0])
    rho_min = np.min(rho_array[:,0])
    with open(args.modelsave_path + "run.txt", "a") as file:
        file.write(f"\n =====>MEAN CC: {rho:.5f}<=====")
    # np.save(modelsave_path +runmodel+'_pred_state.npy', pred_state)
    # np.save(modelsave_path +runmodel+'_ref_state.npy', test_state)
    # np.save(modelsave_path +runmodel+ '_rho_array.npy', rho_array)
    return  pred_state, test_state, rho, rho_min


def main(args):
    setup_seed(111)
    train_start = time.time()
    train_random_indices,train_epochs_loss,val_epochs_loss,best_epoch = train_RK4PIGRU_main(args)
    train_end = time.time()
    train_time_m = (train_end - train_start)/ 60
    with open(args.modelsave_path +"run.txt", "w") as file:
        file.write(f"TRAIN ID:{train_random_indices} \n")
        file.write(f"gru_step = {args.gru_step} \n")
        file.write(f"Time: {train_time_m:.2f} m\n  Best Epoch: {best_epoch}")
    pred_state, test_state,rho_train,rho_min_train = test_RK4PIGRU_main(args,'train',train_random_indices)
    # plot_state(args, pred_state, test_state,'train')
    test_random_indices = [53,45,61,22,51, 7,57,83,29,36,
                           18,41,54, 5,59,84,37,10,60, 9,
                           31,24,55,23, 6,58,13,43,25,74,
                           52,79,82,19,75,48,63,27,38,62,
                           33,73,68,44,34,47,71,81,12,17]
    pred_state, test_state, rho,rho_min   = test_RK4PIGRU_main(args, 'test',test_random_indices)
    return rho,rho_min













