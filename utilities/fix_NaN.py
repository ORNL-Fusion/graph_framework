import netCDF4
import argparse
import numpy

def main(**args):
    for i in range(0, 12):
        with netCDF4.Dataset('{}/result{}.nc'.format(args['directory'], i), 'r+') as result:
            result.variables['kamp'][:] = numpy.where(numpy.isnan(result.variables['kamp'][:]), 0.0, result.variables['kamp'][:])

if __name__ == '__main__':
    command_line_parser = argparse.ArgumentParser()

    command_line_parser.add_argument('-d',
                                     '--directory',
                                     action='store',
                                     required=True,
                                     dest='directory',
                                     help='Directory',
                                     metavar='DIRECTORY')

    args = vars(command_line_parser.parse_args())

#  Remove empty arguments
    for key in [key for key in args if args[key] == None]:
        del args[key]

    main(**args)
