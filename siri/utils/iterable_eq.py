def iterable_eq(a,b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

def iterable_prod(xs):
    out = 1
    for x in xs: out *= x
    return out