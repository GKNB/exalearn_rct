from radical import entk
import os
import argparse, sys, math

class MVP(object):

    def __init__(self):
        self.set_argparse()
        self.set_rmq_env()
        self.am = entk.AppManager(hostname=self.rmq_hostname, port=self.rmq_port, username=self.rmq_username, password=self.rmq_password)

    def set_resource(self, res_desc):
        self.am.resource_desc = res_desc

    def set_rmq_env(self):
        self.rmq_port       = int(os.environ.get('RMQ_PORT', 5672))
        self.rmq_hostname   =     os.environ.get('RMQ_HOSTNAME', '95.217.193.116')
        self.rmq_username   =     os.environ.get('RMQ_USERNAME', 'litan')
        self.rmq_password   =     os.environ.get('RMQ_PASSWORD', 'yG2g7WkufPajVUAq')

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="MVP_EnTK")
        parser.add_argument("--num_rank_sim", "-n", help = "number of MPI ranks")
        parser.add_argument("--num_phase", "-p", help = "number of phases")
        parser.add_argument("--num_epoch", "-e", help = "number of epochs")
        parser.add_argument("--config_root_dir", help = "root directory of configs")
        parser.add_argument("--data_root_dir", help = "root directory of data")
        parser.add_argument("--model_dir", help = "directory of model")
        args = parser.parse_args()
        self.args = args
#        if args.num_rank_sim is None or args.num_phase is None or args.num_epoch is None or args.config_root_dir is None or args.data_root_dir is None or args.model_dir is None:
        if args.num_rank_sim is None or args.num_phase is None or args.config_root_dir is None:
            parser.print_help()
            sys.exit(-1)

    # This is for simulation, return a stage which has a single sim task
    def run_mpi_sweep_hdf5_py(self, phase_idx):

        nproc = int(self.args.num_rank_sim)
        pre_exec_list = [
            "module load conda/2021-09-22",
            "export HDF5_USE_FILE_LOCKING=FALSE",
            "export LD_LIBRARY_PATH=/home/twang3/g2full_theta/GSASII/bindist:$LD_LIBRARY_PATH",
            "echo $LD_LIBRARY_PATH",
            "export OMP_NUM_THREADS=1"
            ]

        t = entk.Task()
        t.pre_exec = pre_exec_list
        t.executable = 'python'
        t.arguments = ['/home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_multi_sym_theta.py',
                       '{}/configs_phase{}/config_1001460_cubic.txt'.format(self.args.config_root_dir, phase_idx),
                       '{}/configs_phase{}/config_1522004_trigonal_part1.txt'.format(self.args.config_root_dir, phase_idx),
                       '{}/configs_phase{}/config_1522004_trigonal_part2.txt'.format(self.args.config_root_dir, phase_idx),
                       '{}/configs_phase{}/config_1531431_tetragonal.txt'.format(self.args.config_root_dir, phase_idx)]
#                       '/lus-projects/CSC249ADCD08/twang/real_work_theta/baseline-rct/sim/configs_phase{}/config_1001460_cubic.txt'.format(phase_idx),
#                       '/lus-projects/CSC249ADCD08/twang/real_work_theta/baseline-rct/sim/configs_phase{}/config_1522004_trigonal_part1.txt'.format(phase_idx),
#                       '/lus-projects/CSC249ADCD08/twang/real_work_theta/baseline-rct/sim/configs_phase{}/config_1522004_trigonal_part2.txt'.format(phase_idx),
#                       '/lus-projects/CSC249ADCD08/twang/real_work_theta/baseline-rct/sim/configs_phase{}/config_1531431_tetragonal.txt'.format(phase_idx)]
        t.post_exec = []
        t.cpu_reqs = {
            'processes': nproc,
            'process_type': 'MPI',
            'threads_per_process': 1,
            'thread_type': 'OpenMP'
        }
        
        s = entk.Stage()
        s.add_tasks(t)

        return s


    # This is for training, return a stage which has a single training task
    def run_mtnetwork_training_horovod_py(self, phase_idx):

        nproc = int(self.args.num_rank)
        
        t = entk.Task()
        t.pre_exec = ['module load conda/2021-09-22',
                      'export OMP_NUM_THREADS=32']
        t.executable = 'python'
        t.arguments = ['/home/twang3/myWork/exalearn_project/run_theta/baseline/ml/mtnetwork-training-horovod.py',
                       '--num_threads=32',
                       '--device cpu',
                       '--epochs={}'.format(self.args.num_epoch),
                       '--phase={}'.format(phase_idx),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir)]
        t.post_exec = []
        t.cpu_reqs = {
            'processes': nproc,
            'process_type': 'MPI',
            'threads_per_process': 64,
            'thread_type': None
            }

        s = entk.Stage()
        s.add_tasks(t)

    def generate_pipeline(self):
        
        p = entk.Pipeline()
        for phase in range(int(self.args.num_phase)):
            s1 = self.run_mpi_sweep_hdf5_py(phase)
            p.add_stages(s1)
#            s2 = self.run_mtnetwork_training_horovod_py(phase)
#            p.add_stages(s2)
        return p

    def run_workflow(self):
        p = self.generate_pipeline()
        self.am.workflow = [p]
        self.am.run()


if __name__ == "__main__":

    mvp = MVP()
    n_nodes = 4
#    n_nodes = math.ceil(float(int(mvp.args.num_rank)/64))
    mvp.set_resource(res_desc = {
        'resource': 'anl.theta',
        'queue'   : 'debug-flat-quad',
        'walltime': 60, #MIN
        'cpus'    : 64 * n_nodes,
        'gpus'    : 0 * n_nodes,
        'project' : 'CSC249ADCD08'
        })
    mvp.generate_pipeline()
    mvp.run_workflow()
