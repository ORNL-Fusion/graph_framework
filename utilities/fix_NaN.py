import netCDF4
import argparse
import numpy

#-------------------------------------------------------------------------------
#  Clean up k imaginary and calculate the power absorbed.
def main(**args):
    with netCDF4.Dataset('{}/bins.nc'.format(args['directory']), 'w') as bin_ref:
        nr = bin_ref.createDimension('nr', 64)
        nz = bin_ref.createDimension('nz', 128)

        bins = bin_ref.createVariable('bins', 'f8', ('nr','nz'))
        bins[:,:] = 0

        for i in range(0, 12):
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
                rbin = numpy.linspace(0.84, 2.5, 65)
                zbin = numpy.linspace(-1.6, 1.6, 129)
                for j in range(64):
                    print(i, j)
                    for k in range(128) :
                        bin_power = numpy.sum(numpy.extract(numpy.logical_and(numpy.greater(r[1:], rbin[j]),
                                                            numpy.logical_and(numpy.less(r[1:], rbin[j + 1]),
                                                                              numpy.logical_and(numpy.greater(z[1:], zbin[k]),
                                                                                                numpy.less(z[1:], zbin[k + 1])))), dpower), axis=0)
                        bins[j, k] += numpy.where(numpy.isnan(bin_power), 0.0, bin_power)

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
