from __future__ import unicode_literals

import xlrd

def toNumber(str):
    ret = float(str);
    if ret < 0 or ret > 255:
        raise RuntimeError()
    return ret


def get_device_table(file_path, check_sample_code=True):
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    nrows = table.nrows

    result = []

    # sample_code   c      cb     t1b   t2b     t1   t2
    #  5            7       8     9     10      11    12
    def interest(row):

        try:
            if check_sample_code:
                code = toNumber(row[5].strip())
            else:
                code = row[5].strip()

            c = toNumber(row[7])
            cb = toNumber(row[8])
            t1b = toNumber(row[9])
            t2b = toNumber(row[10])
            t1 = toNumber(row[11])
            t2 = toNumber(row[12])

            return [code, cb - c, cb, t1b - t1, t1b, t2b - t2, t2b]
        except:
            return []

    for i in range(nrows):
        row = interest(table.row_values(i))
        if len(row) != 0:
            result.append(row)

    return result


print get_device_table('../Downloads/export_result.xls', check_sample_code=False)
