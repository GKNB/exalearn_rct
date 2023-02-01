from radical import entk
import os
import argparse, sys, math

class MVP(object):

    def __init__(self):
        self.set_argparse()
        self._set_rmq()
        self.am = entk.AppManager(hostname=self.rmq_hostname, port=self.rmq_port, username=self.rmq_username, password=self.rmq_password)
        self.p = entk.Pipeline()
        self.s = entk.Stage()

    def _set_rmq(self):
        self.rmq_port = int(os.environ.get('RMQ_PORT', 5672))
        self.rmq_hostname = os.environ.get('RMQ_HOSTNAME', '95.217.193.116')
        self.rmq_username = os.environ.get('RMQ_USERNAME', 'litan')
        self.rmq_password = os.environ.get('RMQ_PASSWORD', 'yG2g7WkufPajVUAq')

    def set_resource(self, res_desc):
        res_desc["schema"] = "local"
        self.am.resource_desc = res_desc

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="MVP_EnTK")
        parser.add_argument("--num_rank", "-n", help="number of MPI ranks")
        args = parser.parse_args()
        self.args = args
        if args.num_rank is None:
            parser.print_help()
            sys.exit(-1)

    def run_mpi_sweep_hdf5_py(self, num_rank):

        nproc_cubic = 64
        pre_exec_list = [
            "module load conda/2021-09-22",
            "export HDF5_USE_FILE_LOCKING=FALSE",
            "export LD_LIBRARY_PATH=/home/twang3/g2full_theta/GSASII/bindist:$LD_LIBRARY_PATH",
            "echo $LD_LIBRARY_PATH",
            "export OMP_NUM_THREADS=1"
            ]

        t_cubic = entk.Task()
        t_cubic.pre_exec = pre_exec_list
        t_cubic.executable = 'python'
        t_cubic.arguments = ['/home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_theta.py', 
                       '/lus-projects/CSC249ADCD08/twang/real_work_theta/baseline-rct/sim/configs/config_1001460_cubic.txt']
        t_cubic.post_exec = []
        t_cubic.cpu_reqs = {
            'processes': nproc_cubic,
            'process_type': 'MPI',
            'threads_per_process': 1,
            'thread_type': 'OpenMP'
        }
        
        nproc       = int(num_rank)

        t_trigonal_p1 = entk.Task()
        t_trigonal_p1.pre_exec = pre_exec_list
        t_trigonal_p1.executable = 'python'
        t_trigonal_p1.arguments = ['/home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_theta.py', 
                       '/lus-projects/CSC249ADCD08/twang/real_work_theta/baseline-rct/sim/configs/config_1522004_trigonal_part1.txt']
        t_trigonal_p1.post_exec = []
        t_trigonal_p1.cpu_reqs = {
            'processes': nproc,
            'process_type': 'MPI',
            'threads_per_process': 1,
            'thread_type': 'OpenMP'
        }

        t_trigonal_p2 = entk.Task()
        t_trigonal_p2.pre_exec = pre_exec_list
        t_trigonal_p2.executable = 'python'
        t_trigonal_p2.arguments = ['/home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_theta.py', 
                       '/lus-projects/CSC249ADCD08/twang/real_work_theta/baseline-rct/sim/configs/config_1522004_trigonal_part2.txt']
        t_trigonal_p2.post_exec = []
        t_trigonal_p2.cpu_reqs = {
            'processes': nproc,
            'process_type': 'MPI',
            'threads_per_process': 1,
            'thread_type': 'OpenMP'
        }

        t_tetragonal = entk.Task()
        t_tetragonal.pre_exec = pre_exec_list
        t_tetragonal.executable = 'python'
        t_tetragonal.arguments = ['/home/twang3/myWork/exalearn-inverse-application/rct_workflows/Cristina/g2dtgen/src/parameter_sweep/mpi_sweep_hdf5_theta.py', 
                       '/lus-projects/CSC249ADCD08/twang/real_work_theta/baseline-rct/sim/configs/config_1531431_tetragonal.txt']
        t_tetragonal.post_exec = []
        t_tetragonal.cpu_reqs = {
            'processes': nproc,
            'process_type': 'MPI',
            'threads_per_process': 1,
            'thread_type': 'OpenMP'
        }

        self.s.add_tasks(t_cubic)
        self.s.add_tasks(t_trigonal_p1)
        self.s.add_tasks(t_trigonal_p2)
        self.s.add_tasks(t_tetragonal)
        self.p.add_stages(self.s)

    def run(self):
        self.am.workflow = [self.p]
        self.am.run()


if __name__ == "__main__":

    mvp = MVP()
    n_nodes = math.ceil(float(int(mvp.args.num_rank)/64))
    mvp.set_resource(res_desc = {
        'resource': 'anl.theta',
        'queue'   : 'debug-flat-quad',
        'walltime': 60, #MIN
        'cpus'    : 64 * n_nodes,
        'gpus'    : 0 * n_nodes,
        'project' : 'CSC249ADCD08'
        })
    mvp.run_mpi_sweep_hdf5_py(num_rank=mvp.args.num_rank)
    mvp.run()
