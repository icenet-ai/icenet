
from preprocess_toolbox.processor import NormalisingChannelProcessor


class SICPreProcessor(NormalisingChannelProcessor):
    def pre_normalisation(self, var_name: str, da: object):
        """

        :param var_name:
        :param da:
        :return:
        """
        if var_name != "siconca":
            raise RuntimeError("OSISAF SIC implementation should be dealing "
                               "with siconca, ")
        # else:
        #     masks = Masks(north=self.north, south=self.south)
        #     return sic_interpolate(da, masks)
        return da
