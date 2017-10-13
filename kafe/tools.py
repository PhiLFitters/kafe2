from __future__ import print_function

def print_dict_recursive(dct, indent_level=0, indent_char='\t', filter_func=lambda x: True):
    for k, v in six.iteritems(dct):
        if isinstance(v, dict):
            print("%s%s" % (indent_char*indent_level, k))
            print_dict_recursive(v, indent_level=indent_level+1, indent_char=indent_char, filter_func=filter_func)
        else:
            if filter_func(v):
                print("%s%s%s%s" % (indent_char * indent_level, k, indent_char, v))
            else:
                #print("*%s%s%s%s" % (indent_char * indent_level, k, indent_char, v))
                pass
