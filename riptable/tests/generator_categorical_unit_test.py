from .groupby_categorical_unit_test_parameters import *


def generate(n):

    PARAMETERS = categorical_parameters()
    epsilon = 0.00003

    def str_filename():
        return "test_sfw_categorical_autotest_aggregated_functions.py"

    # TODO use relative import for riptable
    def str_imports():
        s = "\nfrom .groupby_categorical_unit_test_parameters import *"
        s += "\nimport pandas as pd"
        s += "\nimport riptable as rt"
        s += "\nimport unittest"

        return s

    def str_header():
        return """
        ##                                                                      ##
        #                                                                        #
        #   THIS TEST WAS AUTOGENERATED BY generator_categorical_unit_test.py    #
        #                                                                        #
        ##                                                                      ##   
        """

    def str_agg_params():
        return PARAMETERS.aggs

    def str_agg_params_quoted():
        return '"' + str_agg_params() + '"'

    def str_params():
        return (
            str(PARAMETERS.val_cols)
            + ", "
            + str(PARAMETERS.symbol_ratio + epsilon)[0:4]
            + ", "
            + str_agg_params_quoted()
        )

    def str_name_symratio():
        return "symb_" + str(PARAMETERS.symbol_ratio + epsilon)[0:4].replace(".", "_")

    def str_name_aggs():
        return "aggs_" + str_agg_params()

    def str_name_cols():
        return "ncols_" + str(PARAMETERS.val_cols)

    def str_test_name():
        return str_name_aggs() + "_" + str_name_symratio() + "_" + str_name_cols()

    def str_classname():
        # TODO camel case class names
        return "categorical_test"

    def str_class_header():
        return "\n\nclass " + str_classname() + "(unittest.TestCase):"

    def str_main_tail():
        return '\n\nif __name__ == "__main__":\n\ttester = unittest.main()'

    def str_recycle_off():
        return "\n\t\trt.FastArray._ROFF()"

    def str_recycle_on():
        return "\n\t\trt.FastArray._RON()"

    def str_threads_off():
        return "\n\t\trt.FastArray._TOFF()"

    def str_threads_on():
        return "\n\t\trt.FastArray._TON()"

    def str_standard_test():
        s = ""
        s += "\n\n\tdef test_" + str_test_name() + "(self):"
        s += "\n\t\ttest_class = categorical_base(" + str_params() + ")"
        s += (
            "\n\t\tcat = rt.Categorical(values="
            + "test_class.bin_ids"
            + ", categories= test_class.keys)"
        )
        s += "\n\t\tcat = cat." + str_agg_params() + "(rt.Dataset(test_class.data))"
        # s += '\n\t' + 'test_class = rt.Categorical('
        s += "\n\t\tgb  = pd.DataFrame(test_class.data)"
        s += "\n\t\tgb  = gb.groupby(test_class.bin_ids)." + str_agg_params() + "()"

        s += "\n\t\tfor k,v in test_class.data.items():"
        s += "\n\t\t\tsafe_assert(remove_nan(gb[k]), remove_nan(cat[k]))"

        return s

    def str_nothread_test():
        s = ""
        s += "\n\n\tdef test_nothreads_" + str_test_name() + "(self):"
        s += str_threads_off()
        s += "\n\t\ttest_class = categorical_base(" + str_params() + ")"
        s += (
            "\n\t\tcat = rt.Categorical(values="
            + "test_class.bin_ids"
            + ", categories= test_class.keys)"
        )
        s += "\n\t\tcat = cat." + str_agg_params() + "(rt.Dataset(test_class.data))"
        # s += '\n\t' + 'test_class = rt.Categorical('
        s += "\n\t\tgb  = pd.DataFrame(test_class.data)"
        s += "\n\t\tgb  = gb.groupby(test_class.bin_ids)." + str_agg_params() + "()"

        s += "\n\t\tfor k,v in test_class.data.items():"
        s += "\n\t\t\tsafe_assert(remove_nan(gb[k]), remove_nan(cat[k]))"
        s += str_threads_on()

        return s

    def str_norecycle_test():
        s = ""
        s += "\n\n\tdef test_norecycle_" + str_test_name() + "(self):"
        s += str_recycle_off()
        s += "\n\t\ttest_class = categorical_base(" + str_params() + ")"
        s += (
            "\n\t\tcat = rt.Categorical(values="
            + "test_class.bin_ids"
            + ", categories= test_class.keys)"
        )
        s += "\n\t\tcat = cat." + str_agg_params() + "(rt.Dataset(test_class.data))"
        # s += '\n\t' + 'test_class = rt.Categorical('
        s += "\n\t\tgb  = pd.DataFrame(test_class.data)"
        s += "\n\t\tgb  = gb.groupby(test_class.bin_ids)." + str_agg_params() + "()"

        s += "\n\t\tfor k,v in test_class.data.items():"
        s += "\n\t\t\tsafe_assert(remove_nan(gb[k]), remove_nan(cat[k]))"
        s += str_recycle_on()

        return s

    def str_nothreads_norecycle_test():
        s = ""
        s += "\n\n\tdef test_nothreads_norecycle_" + str_test_name() + "(self):"
        s += str_recycle_off()
        s += str_threads_off()
        s += "\n\t\ttest_class = categorical_base(" + str_params() + ")"
        s += (
            "\n\t\tcat = rt.Categorical(values="
            + "test_class.bin_ids"
            + ", categories= test_class.keys)"
        )
        s += "\n\t\tcat = cat." + str_agg_params() + "(rt.Dataset(test_class.data))"
        # s += '\n\t' + 'test_class = rt.Categorical('
        s += "\n\t\tgb  = pd.DataFrame(test_class.data)"
        s += "\n\t\tgb  = gb.groupby(test_class.bin_ids)." + str_agg_params() + "()"

        s += "\n\t\tfor k,v in test_class.data.items():"
        s += "\n\t\t\tsafe_assert(remove_nan(gb[k]), remove_nan(cat[k]))"
        s += str_recycle_on()
        s += str_threads_on()

        return s

    s = "### Filename = " + str_filename() + "\n"
    s += str_header()
    s += str_imports()

    s += str_class_header()

    for tests_numb in range(0, n):
        s += str_standard_test()
        # s += str_norecycle_test()                   ## remove once switched
        # s += str_nothread_test()                    ## remove once switched
        # s += str_nothreads_norecycle_test()         ## remove once switched

        PARAMETERS.update()

    print(s)

    s += str_main_tail()
    filename = str_filename()
    file = open(filename, "w")
    file.write(s)
    file.close()


if __name__ == "__main__":
    generate(25)
