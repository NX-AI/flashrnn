import sys


def main():
    dict_name = sys.argv[1]
    struct_name = sys.argv[2]

    comment_mode = False

    while True:
        try:
            s = input()
            s = s.strip("\n")
            if comment_mode:
                if "*/" in s:
                    comment = s
                    comment_mode = False
                    print(s, end="\n")
                else:
                    print(s, end="")
                continue

            if "/*" in s:
                sseg = s.split("/*", maxsplit=1)
                non_comment, comment = sseg[0], sseg[1]
                if "*/" not in comment:
                    comment_mode = True
            elif "//" in s:
                non_comment, comment = s.split("//", maxsplit=1)

            decl = non_comment.split(";")[0]
            if "=" in decl:
                var, assign = decl.split("=")
            else:
                var = decl
            var_ = var.split(" ")
            varname_ar = var_[-1]
            if "[" in varname_ar:
                varname, array_size = varname_ar.split("[", maxsplit=1)
                array_size = "[" + array_size
            else:
                varname = varname_ar

            print(
                dict_name
                + '["'
                + varname
                + '"] = '
                + struct_name
                + "."
                + varname
                + ";"
                + "     //"
                + comment,
                end="",
            )
            if not comment_mode:
                print()

        except EOFError:
            break


if __name__ == "__main__":
    main()
