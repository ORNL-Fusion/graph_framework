#-------------------------------------------------------------------------------
##  @file fix_NaN.py
##  @brief Post processes result files.
#-------------------------------------------------------------------------------

import netCDF4
import argparse
import tensorflow

#-------------------------------------------------------------------------------
##  @brief Clean up result files.
##
##  Removes NaN's and noise spikes in the results. Also computes the power
##  absorption and bins the power into a 2D grid.
##
##  @param params[in] args Command line arguments.
#-------------------------------------------------------------------------------
def main(**args):
    with netCDF4.Dataset('{}/bins.nc'.format(args['directory']), 'w') as bin_ref:
        nx = bin_ref.createDimension('nx', args['num_x'])
        ny = bin_ref.createDimension('ny', args['num_y'])
        nz = bin_ref.createDimension('nz', args['num_z'])
        nxp = bin_ref.createDimension('nxp', args['num_x'] + 1)
        nyp = bin_ref.createDimension('nyp', args['num_y'] + 1)
        nzp = bin_ref.createDimension('nzp', args['num_z'] + 1)

        bins = bin_ref.createVariable('bins', 'f8', ('nx','ny','nz'))
        xbin = bin_ref.createVariable('xbins', 'f8', ('nxp'))
        ybin = bin_ref.createVariable('ybins', 'f8', ('nyp'))
        zbin = bin_ref.createVariable('zbins', 'f8', ('nzp'))
        total = 0

        power_bins = tensorflow.zeros((args['num_x'], args['num_y'], args['num_z']), dtype=tensorflow.float64)

        min_x = tensorflow.constant(args['min_x'], dtype=tensorflow.float64)
        max_x = tensorflow.constant(args['max_x'], dtype=tensorflow.float64)
        num_x = tensorflow.constant(args['num_x'])
        min_y = tensorflow.constant(args['min_y'], dtype=tensorflow.float64)
        max_y = tensorflow.constant(args['max_y'], dtype=tensorflow.float64)
        num_y = tensorflow.constant(args['num_y'])
        min_z = tensorflow.constant(args['min_z'], dtype=tensorflow.float64)
        max_z = tensorflow.constant(args['max_z'], dtype=tensorflow.float64)
        num_z = tensorflow.constant(args['num_z'])

        xbins = tensorflow.linspace(min_x, max_x, args['num_x'] + 1)
        ybins = tensorflow.linspace(min_y, max_y, args['num_y'] + 1)
        zbins = tensorflow.linspace(min_z, max_z, args['num_z'] + 1)

        xbin[:] = xbins.numpy()
        ybin[:] = ybins.numpy()
        zbin[:] = zbins.numpy()

        @tensorflow.function
        def mask(x, bin, index):
            return tensorflow.math.logical_and(tensorflow.math.greater_equal(x, bin[index]),
                                               tensorflow.math.less(x, bin[index + 1]))

        @tensorflow.function
        def power_x(x_mask, y_mask, z_mask, dpower, index):
            return tensorflow.map_fn(lambda i: power_y(x_mask[i], y_mask, z_mask, dpower, index, i),
                                     tensorflow.range(num_x),
                                     fn_output_signature=tensorflow.float64)

        @tensorflow.function
        def power_y(x_mask, y_mask, z_mask, dpower, index, i):
            tensorflow.print(index, i)
            return tensorflow.map_fn(lambda j: power_z(x_mask, y_mask[j], z_mask, dpower, i, j),
                                     tensorflow.range(num_y),
                                     fn_output_signature=tensorflow.float64)

        @tensorflow.function
        def power_z(x_mask, y_mask, z_mask, dpower, i, j):
            return tensorflow.map_fn(lambda k: power(x_mask, y_mask, z_mask[k], dpower),
                                     tensorflow.range(num_z),
                                     fn_output_signature=tensorflow.float64)

        @tensorflow.function
        def power(x_mask, y_mask, z_mask, dpower):
            masks = tensorflow.math.logical_and(x_mask,
                                                tensorflow.math.logical_and(y_mask,
                                                                            z_mask))
            return tensorflow.math.reduce_sum(tensorflow.boolean_mask(dpower, masks))

        @tensorflow.function
        def bin_power(x, y, z, dpower, index):
            x_mask = tensorflow.map_fn(lambda i: mask(x, xbins, i),
                                       tensorflow.range(num_x),
                                       fn_output_signature=tensorflow.bool)
            y_mask = tensorflow.map_fn(lambda j: mask(y, ybins, j),
                                       tensorflow.range(num_y),
                                       fn_output_signature=tensorflow.bool)
            z_mask = tensorflow.map_fn(lambda k: mask(z, zbins, k),
                                       tensorflow.range(num_z),
                                       fn_output_signature=tensorflow.bool)
            return power_x(x_mask, y_mask, z_mask, dpower, index)

        for index in range(args['num_files']):
            with netCDF4.Dataset('{}/result{}.nc'.format(args['directory'], index), 'r') as result:
                x = tensorflow.constant(result.variables['x'][:,:,0])
                y = tensorflow.constant(result.variables['y'][:,:,0])
                z = tensorflow.constant(result.variables['z'][:,:,0])
                dpower = tensorflow.constant(result.variables['d_power'][:,:,0])

                total += x.shape[1]
                print(total)
                power_bins += bin_power(x, y, z, dpower, tensorflow.constant(index))

        bins[:,:,:] = power_bins/total

#-------------------------------------------------------------------------------
##  @brief Script entry point.
##
##  Defines command line arguments for.
##  * --directory Directory to search for the result files.
##  * --num_files Number of result files to read.
##  * --num_x     Number of bin points in x.
##  * --min_x     Miniumum x bin.
##  * --max_x     Maximum x bin.
##  * --num_y     Number of bin points in y.
##  * --min_y     Miniumum y bin.
##  * --max_y     Maximum y bin.
##  * --num_z     Number of bin points in z.
##  * --min_z     Miniumum z bin.
##  * --max_z     Maximum z bin.
#-------------------------------------------------------------------------------
if __name__ == '__main__':
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
    command_line_parser.add_argument('-nx',
                                     '--num_x',
                                     action='store',
                                     required=True,
                                     dest='num_x',
                                     help='Number of bin points in x.',
                                     type=int,
                                     metavar='NUM_X')
    command_line_parser.add_argument('-x',
                                     '--min_x',
                                     action='store',
                                     required=True,
                                     dest='min_x',
                                     help='Miniumum x bin.',
                                     type=float,
                                     metavar='MIN_X')
    command_line_parser.add_argument('-mx',
                                     '--max_x',
                                     action='store',
                                     required=True,
                                     dest='max_x',
                                     help='Maximum x bin.',
                                     type=float,
                                     metavar='MAX_X')
    command_line_parser.add_argument('-ny',
                                     '--num_y',
                                     action='store',
                                     required=True,
                                     dest='num_y',
                                     help='Number of bin points in y.',
                                     type=int,
                                     metavar='NUM_Y')
    command_line_parser.add_argument('-y',
                                     '--min_y',
                                     action='store',
                                     required=True,
                                     dest='min_y',
                                     help='Miniumum y bin.',
                                     type=float,
                                     metavar='MIN_Y')
    command_line_parser.add_argument('-my',
                                     '--max_y',
                                     action='store',
                                     required=True,
                                     dest='max_y',
                                     help='Maximum y bin.',
                                     type=float,
                                     metavar='MAX_Y')
    command_line_parser.add_argument('-nz',
                                     '--num_z',
                                     action='store',
                                     required=True,
                                     dest='num_z',
                                     help='Number of bin points in z.',
                                     type=int,
                                     metavar='NUM_Z')
    command_line_parser.add_argument('-z',
                                     '--min_z',
                                     action='store',
                                     required=True,
                                     dest='min_z',
                                     help='Miniumum z bin.',
                                     type=float,
                                     metavar='MIN_Z')
    command_line_parser.add_argument('-mz',
                                     '--max_z',
                                     action='store',
                                     required=True,
                                     dest='max_z',
                                     help='Maximum z bin.',
                                     type=float,
                                     metavar='MAX_Z')

    args = vars(command_line_parser.parse_args())

#  Remove empty arguments
    for key in [key for key in args if args[key] == None]:
        del args[key]

    main(**args)
