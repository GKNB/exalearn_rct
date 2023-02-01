#!/usr/bin/env python3

import radical.pilot as rp
import radical.utils as ru

if __name__ == '__main__':

    # we use a reporter class for nicer output
    report = ru.Reporter(name='radical.pilot')
    report.title('Getting Started (RP version %s)' % rp.version)

    # Create a new session. No need to try/except this: if session creation
    # fails, there is not much we can do anyways...
    session = rp.Session()

    # all other pilot code is now tried/excepted.  If an exception is caught, we
    # can rely on the session object to exist and be valid, and we can thus tear
    # the whole RP stack down via a 'session.close()' call in the 'finally'
    # clause...
    try:

        pmgr   = rp.PilotManager(session=session)
        tmgr   = rp.TaskManager(session=session)

        report.header('submit pilots')

        # Add a PilotManager. PilotManagers manage one or more pilots.

	# Tianle: Here we try to set up them manually!
        pd_init = {'resource'      : 'anl.polaris_interactive',
                   'runtime'       : 30,  # pilot runtime (min)
                   'exit_on_error' : True,
#                   'access_schema' : 'interactive',
                   'cores'         : 1,
                   'gpus'          : 0
                  }
        pdesc = rp.PilotDescription(pd_init)

        # Launch the pilot.
        pilot = pmgr.submit_pilots(pdesc)

        n = 1  # number of tasks to run
        report.header('submit %d tasks' % n)

        # Register the pilot in a TaskManager object.
        tmgr.add_pilots(pilot)

        # Create a workload of tasks.
        # Each task runs '/bin/date'.

        report.progress_tgt(n, label='create')
        tds = list()
        for i in range(0, n):

            # create a new task description, and fill it.
            # Here we don't use dict initialization.
            td = rp.TaskDescription()
            td.stage_on_error = True
            td.executable     = '/bin/date'
            td.cpu_processes  = 1

            tds.append(td)
            report.progress()

        report.progress_done()

        # Submit the previously created task descriptions to the
        # PilotManager. This will trigger the selected scheduler to start
        # assigning tasks to the pilots.
        tmgr.submit_tasks(tds)

        # Wait for all tasks to reach a final state (DONE, CANCELED or FAILED).
        tmgr.wait_tasks()

    except Exception as e:
        # Something unexpected happened in the pilot code above
        report.error('caught Exception: %s\n' % e)
        ru.print_exception_trace()
        raise

    except (KeyboardInterrupt, SystemExit):
        # the callback called sys.exit(), and we can here catch the
        # corresponding KeyboardInterrupt exception for shutdown.  We also catch
        # SystemExit (which gets raised if the main threads exits for some other
        # reason).
        ru.print_exception_trace()
        report.warn('exit requested\n')

    finally:
        # always clean up the session, no matter if we caught an exception or
        # not.  This will kill all remaining pilots.
        report.header('finalize')
        session.close(download=True)

    report.header()


# ------------------------------------------------------------------------------


