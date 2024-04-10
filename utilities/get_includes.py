#-------------------------------------------------------------------------------
##  @file get_includes.py
##  @brief Post processes result files.
#-------------------------------------------------------------------------------
import argparse
import subprocess

#-------------------------------------------------------------------------------
##  @brief Parse the output to get the include directories.
##
##  @params params[in] args Command line arguments.
#-------------------------------------------------------------------------------
def main(**args):
    output = subprocess.run([args['compiler'], '-Wp,-v', '-x', 'c++', '/dev/null', '-fsyntax-only'],
                            capture_output=True,
                            text=True).stderr

    start_capture = False
    includes = ''
    for i, string in enumerate(output.split('\n')):
        if string == '#include <...> search starts here:':
            start_capture = True
            continue
        if string == 'End of search list.':
            break
        if start_capture:
            includes = '{} -I{}'.format(includes, string.split(' ')[1])

    print(includes)

#-------------------------------------------------------------------------------
##  @brief Script entry point.
##
##  This script runs the command
##
##    cc -Wp,-v -x c++ /dev/null -fsyntax-only
##
##  to retrive the include paths for the compiler used.
##
##  Defines command line arguments for.
##  * --compiler Compiler command.
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    command_line_parser = argparse.ArgumentParser()

    command_line_parser.add_argument('-c',
                                     '--compiler',
                                     action='store',
                                     required=True,
                                     dest='compiler',
                                     help='Compiler command.',
                                     metavar='COMPILER')

    args = vars(command_line_parser.parse_args())

#  Remove empty arguments
    for key in [key for key in args if args[key] == None]:
        del args[key]

    main(**args)
