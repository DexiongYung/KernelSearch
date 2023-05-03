import time
import torch
import warnings
import pandas as pd
from botorch.test_functions import Ackley
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize

from parameter_tuning_python.constrained_bayes_opt.trust_region_constrained_bo import TrustRegionConstrainedBayesianOptimization as TRCBO
from parameter_tuning_python.constrained_bayes_opt.trust_region_state import TrustRegionState, update_state

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class TestAckley:

    def __init__(self, num_init, lb=-5, ub=10, con_lb=None, con_ub=None, dim=10):
        """
            To construct the class for testing the convergence on Ackley benchmark.
            :param :dim 'int' dimension of the Ackley function.
            :param :lb/ub 'float' lower and upper bound of parameters for optimization.
            :param :con_lb 'list' of lower bound for each constraint
            :param :con_ub 'list' of upper bound for each constraint
            :param :num_init 'int' number of initial points for initializing surrogate model.
        """

        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.num_init = num_init
        self.con_lb = con_lb
        self.con_ub = con_ub

        assert all(lb <= ub for lb, ub in zip(con_lb, con_ub))

        self.fun = Ackley(dim=dim, negate=True).to(dtype=dtype, device=device)
        self.fun.bounds[0, :].fill_(lb)
        self.fun.bounds[1, :].fill_(ub)

        self.optimal = self.fun.optimal_value

    def ackley(self, X):
        """"
            This is to evaluate Ackley function with raw X
            :param :X torch.tensor with shape (n_sample, dim)
            :return :Y torch.tensor with shape (n_sample, 1)
        """
        # X = torch.tensor(X)
        Y = self.fun(X.to(dtype=dtype))
        return Y[:, None]

    def ackley_contraints(self, X):
        """"
            To evaluate two synthetic constraints
        """
        # X = torch.tensor(X)
        c1 = torch.sum(X, dim=1, keepdim=True)
        c2 = torch.norm(X, p=2, dim=1, keepdim=True) - 5.0
        return torch.cat([c1, c2], dim=-1)

    def weighted_obj_ackley(self, X):
        """"
            To compute weighted objective. If not feasible, set to default kernel_value -20.
            :param : lb - list of lower bound for each constraint
            :param : ub - list of upper bound for each constraint
        """
        # e.g., con_lb = [-5.0, -float("inf")]; con_ub = [0.0, 5.0]
        Y = self.ackley(X)
        con = self.ackley_contraints(X)
        assert len(self.con_lb) == len(self.con_ub) == con.shape[-1]
        dummy_tensor = torch.zeros_like(Y) - 20
        con_lb, con_ub = torch.tensor(self.con_lb).expand_as(con), torch.tensor(self.con_ub).expand_as(con)
        out = torch.where(((con <= con_ub) & (con >= con_lb)).all(dim=-1, keepdim=True), Y, dummy_tensor)
        return out

    def best_weighted_obj(self, X):
        Y_weighted = self.weighted_obj_ackley(X)
        return Y_weighted.max().item()

    def get_initial_points(self, seed=0):
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
        X_norm = sobol.draw(n=self.num_init).to(dtype=dtype)
        X = unnormalize(X_norm, self.fun.bounds)
        Y = self.ackley(X)
        C = self.ackley_contraints(X)
        return X, Y, C

    def prep_ackley_data(self, X_train, Y_train, C_train):
        """
            Prepare the dataframe to construct TR cBO instance.
            X_train, Y_train, C_train is raw (un-normalized) training data.
        """
        data = torch.cat([X_train, Y_train, C_train], dim=1).numpy()

        param_bounds = {}
        for i in range(self.dim):
            param_bounds['x' + str(i)] = (self.lb, self.ub)

        column_names = list(param_bounds.keys()) + ["fx", "c1", "c2"]
        df = pd.DataFrame(columns=column_names)

        training_data_bo = df.append(pd.DataFrame(data, columns=df.columns), ignore_index=True)

        return training_data_bo, param_bounds

    def run_ackley_test(self, num_iter, batch_size, fail_tol, length_min=0.5 ** 14, seed=0,
                        acqf="MonteCarloConstrainedEI", verbose=True):
        """
            Optimize Ackley problem with given number of iteration, batch size, etc.
        """

        # list of best feasible observation
        best_f_list = []

        # interval of constraints
        con_interval = []
        for lb, ub in zip(self.con_lb, self.con_ub):
            con_interval.append([lb, ub])

        # get initial point
        X_train, Y_train, C_train = self.get_initial_points(seed=seed)
        best_f = self.best_weighted_obj(X_train)
        best_f_list.append(best_f)

        # init trust region state
        tr_state = TrustRegionState(length=0.8, failure_tolerance=fail_tol, success_tolerance=6, length_min=length_min)

        for i in range(num_iter):

            if tr_state.restart_triggered:
                break

            t0 = time.time()

            training_data_df, param_bounds = self.prep_ackley_data(X_train, Y_train, C_train)

            model = TRCBO(current_training_data=training_data_df,
                          objective_var='fx',
                          constraint_thresholds={'c1': con_interval[0].copy(), 'c2': con_interval[1].copy()},
                          param_bounds=param_bounds,
                          tr_state=tr_state)

            # t3 = time.time()
            # print('Time to fit the model: ', t3-t0)

            # ['ThompsonSampling', 'BatchConstrainedEI', 'MonteCarloConstrainedEI']
            X_next = model.get_next_batch_sample(batch_size=batch_size, acquisition_func=acqf)

            X_next = torch.from_numpy(X_next)

            # t4 = time.time()
            # print('Time to propose point: ', t4-t3)

            assert X_next.max() <= self.ub and X_next.min() >= self.lb

            # Append the new query points
            Y_next = self.ackley(X_next)
            C_next = self.ackley_contraints(X_next)

            X_train = torch.cat([X_train, X_next], dim=0)
            Y_train = torch.cat([Y_train, Y_next], dim=0)
            C_train = torch.cat([C_train, C_next], dim=0)

            Y_weighted = self.weighted_obj_ackley(X_train)

            tr_state = update_state(state=tr_state, masked_obj=Y_weighted)

            # update progress
            best_f = Y_weighted.max().item()
            best_f_list.append(best_f)

            t1 = time.time()

            if verbose:
                print('Iter: {}, Best feasible objective: {:3.3f}, TR length: {:3.5f}, Elapsed time: {:3.3f}'. \
                      format(i, best_f, tr_state.length, t1 - t0))

        return best_f_list


### qEI ###
num_iter, batch_size, fail_tol = 80, 5, 5

obj_qEI = []

for seed in range(10):
    print(f'_______________{seed} th run_______________')

    ackley_test = TestAckley(num_init=50, lb=-5, ub=10, con_lb=[-5.0, -5.0], con_ub=[0.0, 0.0], dim=10)

    best_f_list = ackley_test.run_ackley_test(num_iter, batch_size, fail_tol, seed=seed, acqf="BatchConstrainedEI",
                                              verbose=False)

    obj_qEI.append(best_f_list)

### TS ###
num_iter, batch_size, fail_tol = 80, 5, 5

obj_TS = []

for seed in range(10):
    print(f'_______________{seed} th run_______________')

    ackley_test = TestAckley(num_init=50, lb=-5, ub=10, con_lb=[-5.0, -5.0], con_ub=[0.0, 0.0], dim=10)

    best_f_list = ackley_test.run_ackley_test(num_iter, batch_size, fail_tol, seed=seed, acqf="ThompsonSampling",
                                              verbose=False)

    obj_TS.append(best_f_list)


### MCEI ###
num_iter, batch_size, fail_tol = 80, 5, 5

obj_MCEI = []

for seed in range(10):
    print(f'_______________{seed} th run_______________')

    ackley_test = TestAckley(num_init=50, lb=-5, ub=10, con_lb=[-5.0, -5.0], con_ub=[0.0, 0.0], dim=10)

    best_f_list = ackley_test.run_ackley_test(num_iter, batch_size, fail_tol, seed=seed, acqf="MonteCarloConstrainedEI",
                                              verbose=False)

    obj_MCEI.append(best_f_list)