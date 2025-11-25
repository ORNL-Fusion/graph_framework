#-------------------------------------------------------------------------------
##  @file fix_NaN.py
##  Post processes result files to remove NaN values.
#-------------------------------------------------------------------------------

import netCDF4
import argparse
import numpy

#-------------------------------------------------------------------------------
##  @brief Clean up result files.
##
##  Removes NaN's and noise spikes in the results. Also computes the power
##  absorption and bins the power into a 2D grid.
##
##  @param[in] args Command line arguments.
#-------------------------------------------------------------------------------
def main(**args):
    with netCDF4.Dataset('{}/bins.nc'.format(args['directory']), 'w') as bin_ref:
        nr = bin_ref.createDimension('nr', 64)
        nz = bin_ref.createDimension('nz', 128)

        bins = bin_ref.createVariable('bins', 'f8', ('nr','nz'))
        bins[:,:] = 0

        rbin = numpy.linspace(args['min_r'],
                              args['max_r'],
                              args['num_r'] + 1)
        zbin = numpy.linspace(args['min_z'],
                              args['max_z'],
                              args['num_z'] + 1)

        for i in range(args['num_files']):
            with netCDF4.Dataset('{}/result{}.nc'.format(args['directory'], i), 'r+') as result:
                result.variables['kamp'][:] = numpy.where(numpy.isnan(result.variables['kamp'][:]), 0.0, result.variables['kamp'][:])
                result.variables['kamp'][:-1] = numpy.where(numpy.abs(result.variables['kamp'][1:] - result.variables['kamp'][:-1]) > 2.0, 0.0, result.variables['kamp'][:-1])
                kampim = result.variables['kamp'][:,:,1]

                x = result.variables['x'][:,:,0]
                y = result.variables['y'][:,:,0]
                z = result.variables['z'][:,:,0]

                kdl = kampim[1:,:]*numpy.sqrt(numpy.power(x[1:] - x[:-1], 2) + numpy.power(y[1:] - y[:-1], 2) + numpy.power(z[1:] - z[:-1], 2));
                power = numpy.append(numpy.ones((1, len(x[1]))), numpy.exp(-2*numpy.cumsum(kdl, axis=0)), axis=0)
                dpower = numpy.abs(power[1:,:] - power[:-1,:])

                r = numpy.sqrt(x*x + y*y)
                for j in range(args['num_r']):
                    print(i, j)
                    for k in range(args['num_z']) :
                        bin_power = numpy.sum(numpy.extract(numpy.logical_and(numpy.greater(r[1:], rbin[j]),
                                                            numpy.logical_and(numpy.less(r[1:], rbin[j + 1]),
                                                                              numpy.logical_and(numpy.greater(z[1:], zbin[k]),
                                                                                                numpy.less(z[1:], zbin[k + 1])))), dpower), axis=0)
                        bins[j, k] += numpy.where(numpy.isnan(bin_power), 0.0, bin_power)

#-------------------------------------------------------------------------------
##  @brief Script entry point.
##
##  Defines command line arguments for.
##  * --directory Directory to search for the result files.
##  * --num_files Number of result files to read.
##  * --num_r     Number of radial bin points.
##  * --min_r     Minimum radial bin.
##  * --max_r     Maximum radial bin.
##  * --num_z     Number of vertical bin points.
##  * --min_z     Minimum vertical bin.
##  * --max_z     Maximum vertical bin.
#-------------------------------------------------------------------------------
if __name__ == '__main__':
##  Argument parser object for command line arguments.
    command_line_parser = argparse.ArgumentParser()

    command_line_parser.add_argument('-d',
                                     '--directory',
                                     action='store',
                                     required=True,
                                     dest='directory',
                                     help='Directory to read result files from.',
                                     metavar='DIRECTORY')
    command_line_parser.add_argument('-n',
                                     '--num_files',
                                     action='store',
                                     required=True,
                                     dest='num_files',
                                     help='Number of result files.',
                                     type=int,
                                     metavar='NUM_FILES')
    command_line_parser.add_argument('-nr',
                                     '--num_r',
                                     action='store',
                                     required=True,
                                     dest='num_r',
                                     help='Number of radial bin points.',
                                     type=int,
                                     metavar='NUM_R')
    command_line_parser.add_argument('-r',
                                     '--min_r',
                                     action='store',
                                     required=True,
                                     dest='min_r',
                                     help='Minimum radial bin.',
                                     type=float,
                                     metavar='MIN_R')
    command_line_parser.add_argument('-mr',
                                     '--max_r',
                                     action='store',
                                     required=True,
                                     dest='max_r',
                                     help='Maximum radial bin.',
                                     type=float,
                                     metavar='MAX_R')
    command_line_parser.add_argument('-j',
                                     '--num_z',
                                     action='store',
                                     required=True,
                                     dest='num_z',
                                     help='Number of vertical bin points.',
                                     type=int,
                                     metavar='NUM_Z')
    command_line_parser.add_argument('-z',
                                     '--min_z',
                                     action='store',
                                     required=True,
                                     dest='min_z',
                                     help='Minimum vertical bin.',
                                     type=float,
                                     metavar='MIN_Z')
    command_line_parser.add_argument('-mz',
                                     '--max_z',
                                     action='store',
                                     required=True,
                                     dest='max_z',
                                     help='Maximum vertical bin.',
                                     type=float,
                                     metavar='MAX_R')

##  The parsed command line arguments.
    args = vars(command_line_parser.parse_args())

#  Remove empty arguments
    for key in [key for key in args if args[key] == None]:
        del args[key]

    main(**args)
