"""
creats a json file where each key can be accessed by the index of a slurm array job.
by accessing the key, the json file can be used to create the input dict for an executable program.
"""

from argparse import ArgumentParser
from itertools import product
import json


def create_dict(*args, **kwargs):
    """
    creates a dict with the product of the args e.g. kwargs.
    use args or kwargs but not both.
    if args is used, args[0] is the key and args[1:] are the values.
    args[1:] should match the key in there order.
    If one input is list(list()) the first dimension will be used for the product.

    if kwargs is used, the key is the key and the value are the values.
    """

    def _create_iterable(arg):

        if isinstance(arg, tuple):
            arg = list(arg)

        for i in range(len(arg)):
            if i == 0:
                continue
            else:
                if isinstance(arg[i], int) or isinstance(arg[i], float) or isinstance(arg[i], str)\
                        or isinstance(arg[i], dict) or isinstance(arg[i], bool):

                    arg[i] = [arg[i]]
                elif isinstance(arg[i], list):
                    pass
                else:
                    raise TypeError(f"{arg[i]} is not a valid type")
        return arg

    def _create_dict(key, arg):
        id_dict = {}

        prod = list(product(*arg))

        i = 0

        while i < len(prod):
            indict = {}
            for k, v in zip(key, prod[i]):
                indict[k] = v

            id_dict[i] = indict
            i += 1

        return id_dict

    if len(args) > 0:

        args = _create_iterable(args)

        return _create_dict(args[0], args[1:])

    elif len(kwargs) > 0:
        arg = _create_iterable(list(kwargs.values()))

        return _create_dict(kwargs.keys(), arg)

    else:
        raise ValueError("No arguments given")


def save_to_file(output_path, dict):
    """
    saves the dict to a json file.
    :param output_path: path to the output file
    :param dict: dict to save
    :return: None
    """
    with open(output_path, "w") as f:
        json.dump(dict, f)
    print(f"saved to {output_path}")


def load_from_file(input_path):
    """
    loads the dict from a json file.
    """
    with open(input_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":

    """
    handle the command line arguments.
    """

    text = "input your parameters for the input file using --iN where N is the number of the parameter.\n" \
           "the first element of the input is the key and the rest are the values.\n" \
           "e.g. --i0 = ex1 1 2   --i1 ex2 3 4  will create a file like \n" \
           "{0:{ex1 : 1,                                         \n" \
           "    ex2 : 3},                                        \n" \
           " 1: {ex1 : 1,                                        \n" \
           "     ex2 : 4},                                       \n" \
           " 2: {ex1 : 2,                                        \n" \
           "     ex2 : 3},                                       \n" \
           " 3: {ex1 : 2,                                        \n" \
           "     ex2 : 4}}                                       \n" \
           "as input arrays out of floats int and str are possible."

    parser = ArgumentParser(description="creates a json file to execute a slurm array job out of it."
                            , usage=text)
    parser.add_argument("-o", "--output", dest="output", help="output path", default="output.json")

    N = 25  # max number of arguments

    for i in range(N):
        parser.add_argument(f"--i{i}", dest=f"i{i}",
                            help="input number{}".format(i),
                            nargs='+', metavar="", default=None)
        # parse unspecified arguments

    args, unknown = parser.parse_known_args()  # parse known arguments

    args_dict = args.__dict__.copy()  # setup the dict for the json file

    args_dict.pop("output")  # remove the output path from the dict as it is not needed in the json file
    # but is needed for the program to save the dict

    for args_key in vars(args):
        attr = getattr(args, args_key)

        # remove the args that are not used in the dict
        if type(attr) is type(None):
            del args_dict[args_key]

        if type(attr) is list:  # if the value is a list, set up the dict for the json file

            new_key = args_dict[args_key][0]  # create new key

            try:
                args_dict[new_key] = [int(i) for i in attr[1:]]  # convert the values to int
            except:
                try:
                    args_dict[new_key] = [float(i) for i in attr[1:]]  # convert the values to float
                except:
                    args_dict[new_key] = attr[1:]  # if the values are not int or float, they are strings

            del args_dict[args_key]  # remove the old key

    # create the dict
    out_dict = create_dict(**args_dict)

    # save the dict
    save_to_file(args.output, out_dict)
