class ConvSettings:
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 dilation=1,
                 stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride

    def compute_internal(self):
        return 2 * self.padding - self.dilation * (self.kernel_size - 1)

    def compute_output(self, I):
        output = (I + self.compute_internal() - 1) / self.stride + 1
        return int(output)

    def copy(self, kernel_size, padding, dilation):
        return ConvSettings(self.in_channels, self.out_channels, kernel_size,
                            padding, dilation, self.stride)

    def is_complete_equivalent(self, other):
        for (self_prop, other_prop) in [
            (self.in_channels, other.in_channels),
            (self.out_channels, other.out_channels),
            (self.kernel_size, other.kernel_size),
            (self.padding, other.padding),
            (self.dilation, other.dilation),
            (self.stride, other.stride),
        ]:
            if self_prop != other_prop:
                return False

        return True

    def is_type_equivalent(self, other):
        if self.stride != other.stride:
            return False

        if self.compute_internal() != other.compute_internal():
            return False

        return True

    def is_instant_equivalent(self, other, I):
        return self.compute_output(I) == other.compute_output(I)

    def generate_type_eq_settings(self, p_limit, d_limit):
        settings = []

        for p in range(p_limit):
            for d in range(1, d_limit):
                k = (2 * p - self.compute_internal()) / d + 1
                if k % 1 or k == 1:
                    continue
                k = int(k)

                new = self.copy(k, p, d)
                if self.is_type_equivalent(new):
                    settings.append(new)
                else:
                    print(self, "NOT EQUAL", new)

        return settings


if __name__ == "__main__":
    import sys
    p_limit, d_limit = int(sys.argv[1]), int(sys.argv[2])
    settings = ConvSettings(0, 0, 3, 1, 1, 1)
    new_settings = settings.generate_type_eq_settings(p_limit, d_limit)
    for ns in new_settings:
        print(ns.kernel_size, ns.padding, ns.dilation, ns.stride,
              "<" if settings.is_complete_equivalent(ns) else "")
