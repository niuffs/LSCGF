import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from lib import utils
from model.pytorch.model import LSCGFModel
from model.pytorch.loss import masked_mae_loss, masked_mape_loss, masked_rmse_loss, masked_mse_loss
import pandas as pd
import os
import time


class LSCGFSupervisor:
    def __init__(self, save_adj_name, args):
        self.args = args
        self.temperature = float(args.temperature)
        self.opt = args.optimizer
        self.max_grad_norm = args.max_grad_norm
        self.ANNEAL_RATE = 0.00003
        self.temp_min = 0.1
        self.save_adj_name = save_adj_name
        self.num_sample = args.num_sample
        self.device = args.device
        # logging.
        self._log_dir = self._get_log_dir(args)
        self._writer = SummaryWriter('runs/' + self._log_dir)
        log_level = args.log_level
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(args.dataset_dir, args.batch_size, args.test_batch_size)
        self.standard_scaler = self._data['scaler']

        ### Feas
        if args.dataset_dir == 'data/METR-LA':
            df = pd.read_hdf('./data/METR-LA/metr-la.h5')
        elif args.dataset_dir == 'data/PEMS-BAY':
            df = pd.read_hdf('./data/PEMS-BAY/pems-bay.h5')
        elif args.dataset_dir == 'data/electricity':
            df = pd.read_csv('./data/electricity/electricity.txt', delimiter=',')
        elif args.dataset_dir == 'data/solar_AL':
            df = pd.read_csv('./data/solar_AL/solar_AL.txt', delimiter=',')
        elif args.dataset_dir == 'data/traffic':
            df = pd.read_csv('./data/traffic/traffic.txt', delimiter=',')
        elif args.dataset_dir == 'data/exchange_rate':
            df = pd.read_csv('./data/exchange_rate/exchange_rate.txt', delimiter=',')

        num_samples = df.shape[0]
        num_train = round(num_samples * 0.7)
        df = df[:num_train].values
        scaler = utils.StandardScaler(mean=df.mean(), std=df.std())
        train_feas = scaler.transform(df)
        self._train_feas = torch.Tensor(train_feas).to(self.device)
        args.num_nodes = df.shape[1]
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.seq_len = args.seq_len  # for the encoder
        self.output_dim = args.output_dim
        self.use_curriculum_learning = args.use_curriculum_learning
        self.horizon = args.horizon  # for the decoder

        # setup model
        LSCGF_model = LSCGFModel(self._train_feas, self.temperature, self._logger, args)
        self.LSCGF_model = LSCGF_model.to(self.device)
        self._logger.info("Model created")

        self._epoch_num = args.epoch
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(args):
        log_dir = args.log_dir
        if log_dir is None:
            batch_size = args.batch_size
            learning_rate = args.base_lr
            max_diffusion_step = args.max_diffusion_step
            num_rnn_layers = args.num_rnn_layers
            rnn_units = args.rnn_units
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = args.horizon
            filter_type = args.filter_type
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'GTS_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = args.base_dir
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        config = {}
        config['model_state_dict'] = self.LSCGF_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models/epo%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self):
        self._setup_graph()
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.LSCGF_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.LSCGF_model = self.LSCGF_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.LSCGF_model(x)
                break

    def train(self, args):
        return self._train(args)

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.LSCGF_model = self.LSCGF_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []
            mapes = []
            # rmses = []
            mses = []
            temp = self.temperature

            l_3 = []
            m_3 = []
            r_3 = []
            l_6 = []
            m_6 = []
            r_6 = []
            l_12 = []
            m_12 = []
            r_12 = []

            for batch_idx, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output, mid_output = self.LSCGF_model(x, temp)

                loss = self._compute_loss(y, output)
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)
                mapes.append(masked_mape_loss(y_pred, y_true).item())
                mses.append(masked_mse_loss(y_pred, y_true).item())
                # rmses.append(masked_rmse_loss(y_pred, y_true).item())
                losses.append(loss.item())

                # Followed the DCRNN TensorFlow Implementation
                l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
                m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
                r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
                l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
                m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
                r_6.append(masked_mse_loss(y_pred[5:6], y_true[5:6]).item())
                l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
                m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
                r_12.append(masked_mse_loss(y_pred[11:12], y_true[11:12]).item())

                if batch_idx % 100 == 1:
                    temp = np.maximum(temp * np.exp(-self.ANNEAL_RATE * batch_idx), self.temp_min)
            mean_loss = np.mean(losses)
            mean_mape = np.mean(mapes)
            mean_rmse = np.sqrt(np.mean(mses))
            # mean_rmse = np.mean(rmses) #another option

            if dataset == 'test':
                # Followed the DCRNN PyTorch Implementation
                message = 'Test: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_loss, mean_mape, mean_rmse)
                self._logger.info(message)

                # Followed the DCRNN TensorFlow Implementation
                message = 'Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_3), np.mean(m_3),
                                                                                           np.sqrt(np.mean(r_3)))
                self._logger.info(message)
                message = 'Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_6), np.mean(m_6),
                                                                                           np.sqrt(np.mean(r_6)))
                self._logger.info(message)
                message = 'Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_12), np.mean(m_12),
                                                                                           np.sqrt(np.mean(r_12)))
                self._logger.info(message)

            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)
            return mean_loss, mean_mape, mean_rmse


    def _train(self, args):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        base_lr = args.base_lr
        steps = args.steps
        patience = args.patience
        epochs = args.epochs
        lr_decay_ratio = args.lr_decay_ratio
        log_every = args.log_every
        save_model = args.save_model
        test_every_n_epochs = args.test_every_n_epochs
        epsilon = args.epsilon

        if self.opt == 'adam':
            optimizer = torch.optim.Adam(self.LSCGF_model.parameters(), lr=base_lr, eps=epsilon)
        elif self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.LSCGF_model.parameters(), lr=base_lr)
        else:
            optimizer = torch.optim.Adam(self.LSCGF_model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=float(lr_decay_ratio))

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        for epoch_num in range(self._epoch_num, epochs):
            print("Num of epoch:", epoch_num)
            self.LSCGF_model = self.LSCGF_model.train()
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []
            start_time = time.time()
            temp = self.temperature

            for batch_idx, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()
                x, y = self._prepare_data(x, y)
                output, mid_output = self.LSCGF_model(x, temp, y, batches_seen)
                if (epoch_num % epochs) == epochs - 1:
                    output, mid_output = self.LSCGF_model(x, temp, y, batches_seen)

                if batches_seen == 0:
                    if self.opt == 'adam':
                        optimizer = torch.optim.Adam(self.LSCGF_model.parameters(), lr=base_lr, eps=epsilon)
                    elif self.opt == 'sgd':
                        optimizer = torch.optim.SGD(self.LSCGF_model.parameters(), lr=base_lr)
                    else:
                        optimizer = torch.optim.Adam(self.LSCGF_model.parameters(), lr=base_lr, eps=epsilon)

                self.LSCGF_model.to(self.device)

                if batch_idx % 100 == 1:
                    temp = np.maximum(temp * np.exp(-self.ANNEAL_RATE * batch_idx), self.temp_min)

                loss = self._compute_loss(y, output)
                losses.append(loss.item())

                self._logger.debug(loss.item())
                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.LSCGF_model.parameters(), self.max_grad_norm)

                optimizer.step()
            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")
            end_time = time.time()


            val_loss, val_mape, val_rmse = self.evaluate(dataset='val', batches_seen=batches_seen)
            end_time2 = time.time()
            self._writer.add_scalar('training loss',
                                    np.mean(losses),
                                    batches_seen)

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, val_mape: {:.4f}, val_rmse: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                    np.mean(losses), val_loss, val_mape, val_rmse,
                                                    lr_scheduler.get_last_lr()[0],
                                                    (end_time - start_time), (end_time2 - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss, test_mape, test_rmse = self.evaluate(dataset='test', batches_seen=batches_seen)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f}, test_mape: {:.4f}, test_rmse: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                    np.mean(losses), test_loss, test_mape, test_rmse,
                                                    lr_scheduler.get_last_lr()[0],
                                                    (end_time - start_time), (end_time2 - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(self.device), y.to(self.device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)
