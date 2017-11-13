import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

class GetData():
    '''
    Load housing data from the housing_path
    optionally adjust sale price for inflation
    optionally do the log transform of the sale price before returning data
      in a transform call
    '''

    def __init__(self, housing_path="../data/AmesHousing.csv",
                adjust_inflation=False,
                log_price=True,
                one_hots=False,
                disable_scaler=False,
                include_PID=False):
        # load in data from csv, when exported from XLS used '~' as the
        # separator
        self.raw_data = pd.read_csv(housing_path, "~")

        # if this is true then adjust sale prices for inflation
        self.adjust_inflation = adjust_inflation

        # do we one-hot categorical or use mapping to ints?
        self.one_hots = one_hots

        # using log price?
        self.log_price = log_price

        # disable scaler during feature evaluation/troubleshooting
        self.disable_scaler = disable_scaler

        # include PID as feature for debugging
        self.include_PID = include_PID

        # self.dummy_data = None

        pd.set_option("display.max_rows",1001)
        pd.set_option("display.max_columns", 1000)

        self.dummy_data = None

    def fit(self):
        # call method to do feature transformations, feature engineering
        if self.one_hots is False:
            self.manipulate_data()
        else:
            self.manipulate_data_w_onehot()

        ##############################
        ### Y Label Manipulations ####
        ##############################

        # if we're using log price set Y to that and drop the normal 'SalePrice'
        if self.log_price:
            # if adjusting for inflation is true
            if self.adjust_inflation:
                # look up how many months since each prop was sold and adjust
                # the sale price to 2010 dollars

                # debug print max price before and after inflation adjustment
                print "     before    max Saleprice:", self.raw_data.loc[:,"SalePrice"].max()

                self.raw_data.loc[:,"real_saleprice"] = self.raw_data.loc[:,"SalePrice"] * self.raw_data.loc[:,"cpi_adjustment"]

                # debug print max price before and after inflation adjustment
                print "     after    max Saleprice:", self.raw_data.loc[:,"SalePrice"].max()

                self.raw_data.loc[:,"log_real_saleprice"] = np.log(self.raw_data.loc[:,"real_saleprice"])

                print "   price after log:", np.exp(self.raw_data.loc[:,"log_real_saleprice"].max())
                self.y = self.raw_data.loc[:,"log_real_saleprice"]

            # otherwise don't adjust log price for inflation
            else:
                self.raw_data.loc[:,"log_saleprice"] = np.log(self.raw_data.loc[:,"SalePrice"])
                self.y = self.raw_data.loc[:,"log_saleprice"]
                print "max price: ", np.exp(self.y.max())
                tprice = self.raw_data.loc[:,"SalePrice"].min()
                tindex = self.raw_data[self.raw_data.loc[:, "SalePrice"] == 1978].loc[:, "PID"]
                print "unadjusted min price:", tprice, "index", tindex
                print "min price: ", np.exp(self.y.min())
                print "PID == 1978:", sum(self.raw_data.loc[:, "TotRms AbvGrd"] == 1978)


        # otherwise don't use log price for Y
        else:
            # if adjusting for inflation is true
            if self.adjust_inflation:
                # look up how many months since each prop was sold and adjust
                # the sale price to 2010 dollars
                self.raw_data.loc[:,"real_saleprice"] = self.raw_data.loc[:,"SalePrice"] * self.raw_data.loc[:,"cpi_adjustment"]
                self.y = self.raw_data.loc[:,"real_saleprice"]
            else:
                # just using straight up saleprices
                self.y = self.raw_data.loc[:, "SalePrice"]

        self.X = self.data

        print self.X.info()

        # print "keys for data_frame:", self.X.keys()

        # scaling on Boolean values appears to be contributing to linear regression
        # explosions?
        scaler = StandardScaler()

        # one-hot columns now stored in self.dummy_data
        if self.disable_scaler is False:
            self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.keys())

        # only scale continuous type variables
        # X_cont = self.X.select_dtypes(include=['float64', "int64"])
        # X_cont_scaled = pd.DataFrame(scaler.fit_transform(X_cont), columns=X_cont.keys())
        #
        # X_discrete = self.X.select_dtypes(include=["uint8", "bool"])

        # print "dummy columns:", self.dummy_data.info()

        # mapping categorical now instead of one-hotting so don't need this
        if self.one_hots is True:
            self.X = pd.concat((self.X, self.dummy_data), axis=1)

        # self.X.loc[:,"PID"] = self.raw_data.loc[:, "PID"]
        # self.X.set_index(self.raw_data.loc[:,"PID"].values)
        # print
        # print
        # print "Data Summary before going into model:"
        # print
        # print self.X.describe()
        # print
        print self.X.info()
        # print
        # print self.X.index.values

        # 372, 945, 2180, 1498, 1533, 2788, 181

        return self

    def transform(self):
        # transform to log prices
        # self.y = np.log(y)

        return self.X, self.y

    def fit_transform(self):
        self.fit()

        # print "len X", len(self.X)
        self.transform()

    def map_column(self, column, col_text):
        '''
        Map a categorical column to numerical values
        use map_dict class to set nan values to 0
        therefore start non-nan values to 1 and higher
        '''
        self.raw_data.loc[:, column].fillna("nan", inplace=True)
        keys = self.raw_data.loc[:, column].unique()

        map_dict = {}

        for idx in range(len(keys)):
            map_dict[keys[idx]] = idx
        #     print idx, keys[idx]

        # train_data['BldgType'].map({'Twnhs':0,'TwnhsE':1,'Duplex':2,'2fmCon':3,'1Fam':4}

        self.data.loc[:, col_text] = self.raw_data.loc[:, column].map(map_dict)

    def dummify(self, column, col_text):
        '''
        take in categorical column and convert it to flagged dummy columns

        Modify pd.get_dummies to drop_first in order to try to fix linear
        regression explosions. This may be happening because of autocorellation
        issue with having one essentially redundant column (k instead of k-1)
        '''
        boink = pd.get_dummies(self.raw_data.loc[:,column], drop_first=True, dummy_na=True).astype(bool)


        keys = boink.keys()
        # make key into a string to handle numberical categories (int)
        new_keys = [col_text + strip_string(key) for key in keys]
        boink.columns = new_keys

        self.dummy_data = pd.concat((self.dummy_data, boink), axis=1).astype(bool)

    def manipulate_data(self):
        '''
        Drop columns, feature engineering, etc

        creates self.X and self.y to eventually be returned by transform
        '''
        # really basic stuff that isn't manipulated at all
        # SalePrice is carried in raw_data and never exists in data DataFrame
        # since price is the Y label and is returned separately

        #############################################################
        # Run: 000/001 minimum viable data                  #########
        #############################################################
        self.data = self.raw_data[["Bedroom AbvGr", "TotRms AbvGrd",
                    "Garage Cars", "Overall Qual", "Lot Area", "Lot Frontage",
                   "Pool Area", "Fireplaces"]]

        self.raw_data.loc[:, "2nd Flr SF"].fillna(0, inplace=True)

        # if include_PID debug is enabled:
        if self.include_PID == True:
            self.data.loc[:,"PID"] = self.raw_data.loc[:,"PID"]

        #############################################################
        # Run: 002 home_sf                                  #########
        #############################################################

        self.data.loc[:,"home_sf"] = self.raw_data.loc[:,"1st Flr SF"] + \
                                        self.raw_data.loc[:,"2nd Flr SF"]

        # fix nans
        self.data.loc[:,"Garage Cars"].fillna(0, inplace=True)

        temp_mean = self.data.loc[:,"Lot Frontage"].mean()
        self.data.loc[:,"Lot Frontage"].fillna(temp_mean, inplace=True)

        #############################################################
        # Run: 003 Overall Qual                             #########
        #############################################################
        self.data.loc[:,"Overall Qual"] = self.raw_data.loc[:,"Overall Qual"]

        #############################################################
        # run: 004 basement stuff                           #########
        #############################################################
        self.map_column("Bsmt Qual", "bsmt_qual_")

        #############################################################
        # run: 005 Bathroom feature manipulation            #########
        #############################################################
        # combine bathroom stats
        self.data.loc[:, "total_baths"] = self.raw_data.loc[:, "Full Bath"] + \
                                            self.raw_data.loc[:, "Half Bath"]

        # if bathrooms == 0 something is probably wrong, add a slight value
        # in case we do a bathroom ratio later
        # self.data.loc[:,"total_baths"] = self.data.loc[:,"total_baths"].mask(self.data.loc[:,"total_baths"] == 0, 0.1)

        #############################################################
        # run: 006 room ratio                               #########
        #############################################################
        self.data.loc[:,"bed_to_room_ratio"] = self.data.loc[:,"Bedroom AbvGr"] \
                                    / (self.data.loc[:,"TotRms AbvGrd"] * 1.0)

        #############################################################
        # run: 007 yrs_since_update                         #########
        #############################################################
        self.data.loc[:,"yrs_since_update"] = 2010 -self.raw_data["Year Remod/Add"]

        #############################################################
        # run: 008 overall cond flags                       #########
        #############################################################
        self.map_column("Overall Cond", "oc_")

        #############################################################
        # run: 009 years since sold                         #########
        #############################################################
        # figure out how long since sale (in years)
        # StandardScaler was making LinearRegression explode because of this feature
        # which is sort of more categorical anyways
        self.raw_data.loc[:,"yrs_since_sold"] = 2010 - self.raw_data.loc[:,"Yr Sold"]
        self.map_column("yrs_since_sold", "yrs_since_sold_")

        #############################################################
        # run: 011 has_2nd_flr flag                         #########
        #############################################################
        # self.data[:,"has_2nd_flr"] = self.raw_data.loc[:,"2nd Flr SF"] != 0
        # self.data[:,"has_2nd_flr"] = self.raw_data.loc[:,"2nd Flr SF"] != 0
        # self.dummy_data = pd.DataFrame((self.raw_data.loc[:,"2nd Flr SF"] != 0), columns=["has_2nd_flr"]).astype(bool)
        # [:,"has_2nd_flr"] = self.raw_data.loc[:,"2nd Flr SF"] != 0

        # self.data.loc[:,"PID"] = self.raw_data.loc[:,"PID"]

        # #############################################################
        # # pool stuff flag                                   #########
        # #############################################################
        #
        # has_pool = np.array(self.data.loc[:,"Pool Area"] > 0)
        # has_pool_df = pd.DataFrame(has_pool, columns=["has_pool"])
        # self.data = pd.concat((self.data, has_pool_df), axis=1)

        #############################################################
        # Stuff to use for inflation adjustments            #########
        #############################################################

        # do the same for raw_data since we'll prob use that as the basis
        # for calculating inflation adjustment as Mo Sold and Months since sale
        # don't see super useful for the actual model data (but will be eval-ed)
        self.raw_data.loc[:,"yrs_since_sold"] = 2010 - self.raw_data.loc[:,"Yr Sold"]

        self.raw_data.loc[:, "mon_since_sold"] = self.raw_data.apply(\
            lambda x: figure_months(x["yrs_since_sold"], x["Mo Sold"]), axis=1)

        # print "mon_since_sold: ", sorted(self.raw_data.loc[:,"mon_since_sold"].unique())
        # mon_since_sold:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        # 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        # 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        # 49, 50, 51, 52, 53, 54]

        np_cpi = load_cpi_data()

        self.raw_data.loc[:, "cpi_adjustment"] = self.raw_data.apply(lambda x: simple_inflation_adjuster(x["mon_since_sold"], np_cpi), axis=1)
        # print "cpi_adjustment:", sorted(self.raw_data.loc[:, "cpi_adjustment"].unique())
        # cpi_adjustment: [1.0, 1.0020171457387794, *snip*,
        #                 1.1048209783156833, 1.1092486132123047]

        #############################################################
        # run: 012 bring in months since sold and mo sold   #########
        #############################################################
        # bring months since sold to model
        self.data.loc[:,"mon_since_sold"] = self.raw_data.loc[:, "mon_since_sold"]

        # bring in what month sold, also
        self.data.loc[:, "month_sold"] = self.raw_data.loc[:,"Mo Sold"]

        #############################################################
        # run: 013 Zoning info                              #########
        #############################################################
        # zoning_cols = ["zoning_rl", "zoning_rh", "zoning_fv", "zoning_rm",
        #                 "zoning_c_all", "zoning_i_all", "zoning_a_all"]

        self.map_column("MS Zoning", "zoning_")
        # #raw_data.loc[:,"MS Zoning"].unique()
        # # array(['RL', 'RH', 'FV', 'RM', 'C (all)', 'I (all)', 'A (agr)'], dtype=object)
        # boink = pd.get_dummies(self.raw_data.loc[:,"MS Zoning"])
        #
        # boink.columns = ["zoning_rl", "zoning_rh", "zoning_fv", "zoning_rm", "zoning_c_all", "zoning_i_all", "zoning_a_all"]
        #
        # self.data = pd.concat((self.data, boink), axis=1)

        #############################################################
        # run: 014 Roof stuff                               #########
        #############################################################
        self.map_column("Roof Style", "roof_")

        #############################################################
        # run: 016 Kitchen qual                             #########
        #############################################################
        self.map_column("Kitchen Qual", "kqual_")

        #############################################################
        # run: 017 Neighborhood, lot shape flags            #########
        #############################################################
        self.map_column("Neighborhood", "neigh_")

        self.map_column("Lot Shape", "lot_shape_")

        #############################################################
        # run: 018 has_huge_liv_area                        #########
        #############################################################
        self.data.loc[:,"has_huge_liv_area"] = (self.data["home_sf"] > 4000).astype(bool)
        self.data.loc[:,"has_tiny_liv_area"] = (self.data["home_sf"] < 500).astype(bool)

        #############################################################
        # run: 019 log transformation of living area        #########
        #############################################################
        self.data.loc[:,"log_home_sf"] = np.log(self.data["home_sf"])
        self.data.pop("home_sf")


        #############################################################
        # run: 020 Condition types, road sizes near house?  #########
        #############################################################
        self.map_column("Condition 1", "cond_1_")
        self.map_column("Condition 2", "cond_2_")

        #############################################################
        # run: 021 fireplace stuff, sale type info          #########
        #############################################################
        self.map_column("Fireplace Qu", "fp_qual_")

        #############################################################
        # sale type    and condition                        #########
        #############################################################
        self.map_column("Sale Condition", "sale_cond_")
        self.map_column("Sale Type", "sale_type_")

        #############################################################
        # run: 022 house style, land stuff                  #########
        #############################################################
        self.map_column("House Style", "house_style_")

        self.map_column("Land Contour", "land_cont_")
        self.map_column("Lot Config", "lot_conf_")
        self.map_column("Land Slope", "land_slope_")

        #############################################################
        # run: 023 exterior stuff                           #########
        #############################################################

        self.map_column("Exterior 1st", "ext_1_")
        self.map_column("Exterior 2nd", "ext_2_")

        self.map_column("Exter Qual", "ext_qual_")
        self.map_column("Exter Cond", "ext_cond_")

        self.map_column("Foundation", "foundation_")

        self.raw_data.loc[:, "Mas Vnr Type"].fillna("None", inplace=True)
        self.map_column("Mas Vnr Type", "mas_vnr_typ_")
        self.raw_data.loc[:,"Mas Vnr Area"].fillna(0, inplace=True)
        self.data.loc[:,"Mas Vnr Area"] = self.raw_data.loc[:, "Mas Vnr Area"]

        #############################################################
        # run: 024 Much more basement stuff                 #########
        #############################################################
        # Actually I think pandas dummify will handle nan's ok
        # self.raw_data.loc[:,"Bsmt Cond"].fillna("TA", inplace=True)
        self.map_column("Bsmt Cond", "bsmt_cond_")

        self.map_column("Bsmt Exposure", "bsmt_expos_")

        self.map_column("BsmtFin Type 1", "bsmt_fin_typ1_")

        # seems like there is one nan value but can replace with 0
        # probably the nan doesn't have a basement?
        self.raw_data.loc[:, "BsmtFin SF 1"].fillna(0, inplace=True)
        self.data.loc[:,"bsmtfin_sf_1"] = self.raw_data.loc[:, "BsmtFin SF 1"]

        self.map_column("BsmtFin Type 2", "bsmt_fin_typ2_")

        self.raw_data.loc[:,"BsmtFin SF 2"].fillna(0, inplace=True)
        self.data.loc[:,"bsmtfin_sf_2"] = self.raw_data.loc[:, "BsmtFin SF 2"]

        self.raw_data.loc[:,"Bsmt Unf SF"].fillna(0, inplace=True)
        self.data.loc[:,"bsmtunf_sf"] = self.raw_data.loc[:, "Bsmt Unf SF"]

        self.raw_data.loc[:,"Total Bsmt SF"].fillna(0, inplace=True)
        self.data.loc[:,"log_total_bsmt_sf"] = np.log(self.raw_data.loc[:,"Total Bsmt SF"] + 1.0)

        #############################################################
        # run: 025 Bldg type, roof_matl                     #########
        #############################################################
        self.map_column("Bldg Type", "bldg_type_")
        self.map_column("Roof Matl", "roof_matl_")

        #############################################################
        # run: 026 heating, central air, electrical onehots #########
        #############################################################
        self.map_column("Heating", "heating_")
        self.map_column("Heating QC", "heat_qc_")
        self.map_column("Central Air", "ac_")
        self.map_column("Electrical", "electrical_")

        #############################################################
        # run: 027 low qual SF, bsmt baths, kitchens, functional ####
        #############################################################
        self.data.loc[:,"Low Qual Fin SF"] = self.raw_data.loc[:,"Low Qual Fin SF"]

        self.raw_data.loc[:, "Bsmt Full Bath"].fillna(0, inplace=True)
        self.data.loc[:,"Bsmt Full Bath"] = self.raw_data.loc[:, "Bsmt Full Bath"]

        self.raw_data.loc[:, "Bsmt Half Bath"].fillna(0, inplace=True)
        self.data.loc[:,"Bsmt Half Bath"] = self.raw_data.loc[:, "Bsmt Half Bath"]

        self.data.loc[:, "Kitchen AbvGr"] = self.raw_data.loc[:, "Kitchen AbvGr"]

        self.map_column("Functional", "funct_")

        #############################################################
        # run: 028 tons of garage stuff and also paved_drive_    ####
        #############################################################
        self.map_column("Garage Type", "garage_type_")

        # for some reason there is a value that is 2207 which is like 200 years
        # in the future. Fix all future values and replace with the mean
        mean_yr = int(self.raw_data.loc[:, "Garage Yr Blt"].mean())

        self.raw_data[self.raw_data.loc[:, "Garage Yr Blt"] > 2010] = mean_yr

        # self.raw_data[self.raw_data.loc[:, "Garage Finish"] == 1978] = "Unf"
        # replace nan's with the mean too, otherwise it will be very imbalanced?

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!! Doing something dumb? Revisit later !!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.raw_data.loc[:, "Garage Yr Blt"].fillna(mean_yr, inplace=True)

        self.data.loc[:,"garage_yr_blt"] = self.raw_data.loc[:, "Garage Yr Blt"]

        self.map_column("Garage Qual", "garage_qual_")

        self.map_column("Garage Cond", "garage_cond_")

        self.map_column("Paved Drive", "paved_drive_")

        #############################################################
        # run: 029 deck/porch stuff                              ####
        #############################################################
        self.data.loc[:, "log_wood_deck_sf"] = np.log(self.raw_data.loc[:, "Wood Deck SF"] + 1.0)
        self.data.loc[:, "log_open_porch_sf"] = np.log(self.raw_data.loc[:, "Open Porch SF"] + 1.0)
        self.data.loc[:, "log_enc_porch_sf"] = np.log(self.raw_data.loc[:, "Enclosed Porch"] + 1.0)
        self.data.loc[:, "log_3ssn_porch_sf"] = np.log(self.raw_data.loc[:, "3Ssn Porch"] + 1.0)
        self.data.loc[:, "log_scrn_porch_sf"] = np.log(self.raw_data.loc[:, "Screen Porch"] + 1.0)

        # self.data.loc[:, "Wood Deck SF"] = self.raw_data.loc[:, "Wood Deck SF"]
        # self.data.loc[:, "Open Porch SF"] = self.raw_data.loc[:, "Open Porch SF"]
        # self.data.loc[:, "Enclosed Porch"] = self.raw_data.loc[:, "Enclosed Porch"]
        # self.data.loc[:, "3Ssn Porch"] = self.raw_data.loc[:, "3Ssn Porch"]
        # self.data.loc[:, "Screen Porch"] = self.raw_data.loc[:, "Screen Porch"]
        # self.map_column("Misc Feature", "misc_feat_")

        #############################################################
        # run: 031 synthetic stuff                               ####
        #############################################################
        self.data.loc[:, "bed_to_bath_ratio"] = self.data.loc[:,"Bedroom AbvGr"] / ((self.data.loc[:,"total_baths"] + 1.0) * 1.0)
        self.data.loc[:, "log_lot_area"] = np.log(self.data.loc[:, "Lot Area"] + np.exp(1.0))
        self.data.pop("Lot Area")
        self.data.loc[:, "house_to_land_ratio"] = self.data.loc[:, "log_home_sf"] / self.data.loc[:,"log_lot_area"]

        #############################
        #############################
        ##### JUNKED FEATURES #######
        #############################
        #############################

        ################################
        ### Dead stuff from stage 28 ###
        ################################

        # # Garage finish has a column named (int) 1978 which doesn't seem to make sense
        # # self.raw_data[self.raw_data.loc[:, "Garage Finish"] == 1978] = "Unf"
        # self.map_column("Garage Finish", "garage_fin_")
        #
        # self.raw_data.loc[:, "Garage Area"].fillna(0, inplace=True)
        #
        # # has no garage
        # self.dummy_data.loc[:,"has_garage"] = (self.raw_data["Garage Area"] < 1.0).astype(bool)
        #
        # # take the log of the garage area (TODO fuzz some to see if score improves
        # # because there is a big class of 0ft^2, although might be fixed with
        # # dummy column has_garage)
        # self.raw_data.loc[:, "Garage Area"].apply(lambda x: np.log(x + 1.0))
        # self.data.loc[:,"log_garage_area"] = self.raw_data.loc[:, "Garage Area"]


        #
        # # amount of lot used ratio? lot / square_feet
        # # self.data.loc[:,"lot_home_area_ratio"] = self.raw_data.loc[:,"Lot Area"] \
        # #                         / (self.data.loc[:, "home_sf"] * 1.0)
        #

    def manipulate_data_w_onehot(self):
        '''
        Drop columns, feature engineering, etc

        creates self.X and self.y to eventually be returned by transform
        '''
        # really basic stuff that isn't manipulated at all
        # SalePrice is carried in raw_data and never exists in data DataFrame
        # since price is the Y label and is returned separately

        #############################################################
        # Run: 000/001 minimum viable data                  #########
        #############################################################
        self.data = self.raw_data[["Bedroom AbvGr", "TotRms AbvGrd",
                    "Garage Cars", "Overall Qual", "Lot Area", "Lot Frontage",
                   "Pool Area", "Fireplaces"]]

        self.raw_data.loc[:, "2nd Flr SF"].fillna(0, inplace=True)

        # if include_PID debug is enabled:
        if self.include_PID == True:
            self.data.loc[:,"PID"] = self.raw_data.loc[:,"PID"]


        #############################################################
        # Run: 002 home_sf                                  #########
        #############################################################

        self.data.loc[:,"home_sf"] = self.raw_data.loc[:,"1st Flr SF"] + \
                                        self.raw_data.loc[:,"2nd Flr SF"]

        # fix nans
        self.data.loc[:,"Garage Cars"].fillna(0, inplace=True)

        temp_mean = self.data.loc[:,"Lot Frontage"].mean()
        self.data.loc[:,"Lot Frontage"].fillna(temp_mean, inplace=True)

        #############################################################
        # run: 011 has_2nd_flr flag                         #########
        #############################################################
        self.dummy_data = pd.DataFrame((self.raw_data.loc[:,"2nd Flr SF"] != 0), columns=["has_2nd_flr"]).astype(bool)

        #############################################################
        # Run: 003 Overall Qual                             #########
        #############################################################
        self.data.loc[:,"Overall Qual"] = self.raw_data.loc[:,"Overall Qual"]

        #############################################################
        # run: 004 basement stuff                           #########
        #############################################################
        self.dummify("Bsmt Qual", "bsmt_qual_")

        #############################################################
        # run: 005 Bathroom feature manipulation            #########
        #############################################################
        # combine bathroom stats
        self.data.loc[:, "total_baths"] = self.raw_data.loc[:, "Full Bath"] + \
                                            self.raw_data.loc[:, "Half Bath"]

        # if bathrooms == 0 something is probably wrong, add a slight value
        # in case we do a bathroom ratio later
        # self.data.loc[:,"total_baths"] = self.data.loc[:,"total_baths"].mask(self.data.loc[:,"total_baths"] == 0, 0.1)

        #############################################################
        # run: 006 room ratio                               #########
        #############################################################
        self.data.loc[:,"bed_to_room_ratio"] = self.data.loc[:,"Bedroom AbvGr"] \
                                    / (self.data.loc[:,"TotRms AbvGrd"] * 1.0)

        #############################################################
        # run: 007 yrs_since_update                         #########
        #############################################################
        self.data.loc[:,"yrs_since_update"] = 2010 -self.raw_data["Year Remod/Add"]

        #############################################################
        # run: 008 overall cond flags                       #########
        #############################################################
        self.dummify("Overall Cond", "oc_")

        #############################################################
        # run: 009 years since sold                         #########
        #############################################################
        # figure out how long since sale (in years)
        # StandardScaler was making LinearRegression explode because of this feature
        # which is sort of more categorical anyways
        self.raw_data.loc[:,"yrs_since_sold"] = 2010 - self.raw_data.loc[:,"Yr Sold"]
        self.dummify("yrs_since_sold", "yrs_since_sold_")

        # self.data.loc[:,"PID"] = self.raw_data.loc[:,"PID"]

        # #############################################################
        # # pool stuff flag                                   #########
        # #############################################################
        #
        # has_pool = np.array(self.data.loc[:,"Pool Area"] > 0)
        # has_pool_df = pd.DataFrame(has_pool, columns=["has_pool"])
        # self.data = pd.concat((self.data, has_pool_df), axis=1)

        #############################################################
        # Stuff to use for inflation adjustments            #########
        #############################################################

        # do the same for raw_data since we'll prob use that as the basis
        # for calculating inflation adjustment as Mo Sold and Months since sale
        # don't see super useful for the actual model data (but will be eval-ed)
        self.raw_data.loc[:,"yrs_since_sold"] = 2010 - self.raw_data.loc[:,"Yr Sold"]

        self.raw_data.loc[:, "mon_since_sold"] = self.raw_data.apply(\
            lambda x: figure_months(x["yrs_since_sold"], x["Mo Sold"]), axis=1)

        # print "mon_since_sold: ", sorted(self.raw_data.loc[:,"mon_since_sold"].unique())
        # mon_since_sold:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        # 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        # 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        # 49, 50, 51, 52, 53, 54]

        np_cpi = load_cpi_data()

        self.raw_data.loc[:, "cpi_adjustment"] = self.raw_data.apply(lambda x: simple_inflation_adjuster(x["mon_since_sold"], np_cpi), axis=1)
        # print "cpi_adjustment:", sorted(self.raw_data.loc[:, "cpi_adjustment"].unique())
        # cpi_adjustment: [1.0, 1.0020171457387794, *snip*,
        #                 1.1048209783156833, 1.1092486132123047]

        #############################################################
        # run: 012 bring in months since sold and mo sold   #########
        #############################################################
        # bring months since sold to model
        self.data.loc[:,"mon_since_sold"] = self.raw_data.loc[:, "mon_since_sold"]

        # bring in what month sold, also
        self.data.loc[:, "month_sold"] = self.raw_data.loc[:,"Mo Sold"]

        #############################################################
        # run: 013 Zoning info                              #########
        #############################################################
        # zoning_cols = ["zoning_rl", "zoning_rh", "zoning_fv", "zoning_rm",
        #                 "zoning_c_all", "zoning_i_all", "zoning_a_all"]

        self.dummify("MS Zoning", "zoning_")
        # #raw_data.loc[:,"MS Zoning"].unique()
        # # array(['RL', 'RH', 'FV', 'RM', 'C (all)', 'I (all)', 'A (agr)'], dtype=object)
        # boink = pd.get_dummies(self.raw_data.loc[:,"MS Zoning"])
        #
        # boink.columns = ["zoning_rl", "zoning_rh", "zoning_fv", "zoning_rm", "zoning_c_all", "zoning_i_all", "zoning_a_all"]
        #
        # self.data = pd.concat((self.data, boink), axis=1)

        #############################################################
        # run: 014 Roof stuff                               #########
        #############################################################
        self.dummify("Roof Style", "roof_")

        #############################################################
        # run: 016 Kitchen qual                             #########
        #############################################################
        self.dummify("Kitchen Qual", "kqual_")

        #############################################################
        # run: 017 Neighborhood, lot shape flags            #########
        #############################################################
        self.dummify("Neighborhood", "neigh_")

        self.dummify("Lot Shape", "lot_shape_")

        #############################################################
        # run: 018 has_huge_liv_area                        #########
        #############################################################
        self.data.loc[:,"has_huge_liv_area"] = (self.data["home_sf"] > 4000).astype(bool)
        self.data.loc[:,"has_tiny_liv_area"] = (self.data["home_sf"] < 500).astype(bool)

        #############################################################
        # run: 019 log transformation of living area        #########
        #############################################################
        self.data.loc[:,"log_home_sf"] = np.log(self.data["home_sf"])
        self.data.pop("home_sf")


        #############################################################
        # run: 020 Condition types, road sizes near house?  #########
        #############################################################
        self.dummify("Condition 1", "cond_1_")
        self.dummify("Condition 2", "cond_2_")

        #############################################################
        # run: 021 fireplace stuff, sale type info          #########
        #############################################################
        self.dummify("Fireplace Qu", "fp_qual_")

        #############################################################
        # sale type    and condition                        #########
        #############################################################
        self.dummify("Sale Condition", "sale_cond_")
        self.dummify("Sale Type", "sale_type_")

        #############################################################
        # run: 022 house style, land stuff                  #########
        #############################################################
        self.dummify("House Style", "house_style_")

        self.dummify("Land Contour", "land_cont_")
        self.dummify("Lot Config", "lot_conf_")
        self.dummify("Land Slope", "land_slope_")

        #############################################################
        # run: 023 exterior stuff                           #########
        #############################################################

        self.dummify("Exterior 1st", "ext_1_")
        self.dummify("Exterior 2nd", "ext_2_")

        self.dummify("Exter Qual", "ext_qual_")
        self.dummify("Exter Cond", "ext_cond_")

        self.dummify("Foundation", "foundation_")

        self.raw_data.loc[:, "Mas Vnr Type"].fillna("None", inplace=True)
        self.dummify("Mas Vnr Type", "mas_vnr_typ_")
        self.raw_data.loc[:,"Mas Vnr Area"].fillna(0, inplace=True)
        self.data.loc[:,"Mas Vnr Area"] = self.raw_data.loc[:, "Mas Vnr Area"]

        #############################################################
        # run: 024 Much more basement stuff                 #########
        #############################################################
        # Actually I think pandas dummify will handle nan's ok
        # self.raw_data.loc[:,"Bsmt Cond"].fillna("TA", inplace=True)
        self.dummify("Bsmt Cond", "bsmt_cond_")

        self.dummify("Bsmt Exposure", "bsmt_expos_")

        self.dummify("BsmtFin Type 1", "bsmt_fin_typ1_")

        # seems like there is one nan value but can replace with 0
        # probably the nan doesn't have a basement?
        self.raw_data.loc[:, "BsmtFin SF 1"].fillna(0, inplace=True)
        self.data.loc[:,"bsmtfin_sf_1"] = self.raw_data.loc[:, "BsmtFin SF 1"]

        self.dummify("BsmtFin Type 2", "bsmt_fin_typ2_")

        self.raw_data.loc[:,"BsmtFin SF 2"].fillna(0, inplace=True)
        self.data.loc[:,"bsmtfin_sf_2"] = self.raw_data.loc[:, "BsmtFin SF 2"]

        self.raw_data.loc[:,"Bsmt Unf SF"].fillna(0, inplace=True)
        self.data.loc[:,"bsmtunf_sf"] = self.raw_data.loc[:, "Bsmt Unf SF"]

        self.raw_data.loc[:,"Total Bsmt SF"].fillna(0, inplace=True)
        self.data.loc[:,"log_total_bsmt_sf"] = np.log(self.raw_data.loc[:,"Total Bsmt SF"] + 1.0)

        #############################################################
        # run: 025 Bldg type, roof_matl                     #########
        #############################################################
        self.dummify("Bldg Type", "bldg_type_")
        self.dummify("Roof Matl", "roof_matl_")

        #############################################################
        # run: 026 heating, central air, electrical onehots #########
        #############################################################
        self.dummify("Heating", "heating_")
        self.dummify("Heating QC", "heat_qc_")
        self.dummify("Central Air", "ac_")
        self.dummify("Electrical", "electrical_")

        #############################################################
        # run: 027 low qual SF, bsmt baths, kitchens, functional ####
        #############################################################
        self.data.loc[:,"Low Qual Fin SF"] = self.raw_data.loc[:,"Low Qual Fin SF"]

        self.raw_data.loc[:, "Bsmt Full Bath"].fillna(0, inplace=True)
        self.data.loc[:,"Bsmt Full Bath"] = self.raw_data.loc[:, "Bsmt Full Bath"]

        self.raw_data.loc[:, "Bsmt Half Bath"].fillna(0, inplace=True)
        self.data.loc[:,"Bsmt Half Bath"] = self.raw_data.loc[:, "Bsmt Half Bath"]

        self.data.loc[:, "Kitchen AbvGr"] = self.raw_data.loc[:, "Kitchen AbvGr"]

        self.dummify("Functional", "funct_")

        #############################################################
        # run: 028 tons of garage stuff and also paved_drive_    ####
        #############################################################
        self.dummify("Garage Type", "garage_type_")

        # for some reason there is a value that is 2207 which is like 200 years
        # in the future. Fix all future values and replace with the mean
        mean_yr = int(self.raw_data.loc[:, "Garage Yr Blt"].mean())

        self.raw_data[self.raw_data.loc[:, "Garage Yr Blt"] > 2010] = mean_yr

        # self.raw_data[self.raw_data.loc[:, "Garage Finish"] == 1978] = "Unf"
        # replace nan's with the mean too, otherwise it will be very imbalanced?

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!! Doing something dumb? Revisit later !!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.raw_data.loc[:, "Garage Yr Blt"].fillna(mean_yr, inplace=True)

        self.data.loc[:,"garage_yr_blt"] = self.raw_data.loc[:, "Garage Yr Blt"]

        self.dummify("Garage Qual", "garage_qual_")

        self.dummify("Garage Cond", "garage_cond_")

        self.dummify("Paved Drive", "paved_drive_")

        #############################################################
        # run: 029 deck/porch stuff                              ####
        #############################################################
        self.data.loc[:, "log_wood_deck_sf"] = np.log(self.raw_data.loc[:, "Wood Deck SF"] + 1.0)
        self.data.loc[:, "log_open_porch_sf"] = np.log(self.raw_data.loc[:, "Open Porch SF"] + 1.0)
        self.data.loc[:, "log_enc_porch_sf"] = np.log(self.raw_data.loc[:, "Enclosed Porch"] + 1.0)
        self.data.loc[:, "log_3ssn_porch_sf"] = np.log(self.raw_data.loc[:, "3Ssn Porch"] + 1.0)
        self.data.loc[:, "log_scrn_porch_sf"] = np.log(self.raw_data.loc[:, "Screen Porch"] + 1.0)

        # self.data.loc[:, "Wood Deck SF"] = self.raw_data.loc[:, "Wood Deck SF"]
        # self.data.loc[:, "Open Porch SF"] = self.raw_data.loc[:, "Open Porch SF"]
        # self.data.loc[:, "Enclosed Porch"] = self.raw_data.loc[:, "Enclosed Porch"]
        # self.data.loc[:, "3Ssn Porch"] = self.raw_data.loc[:, "3Ssn Porch"]
        # self.data.loc[:, "Screen Porch"] = self.raw_data.loc[:, "Screen Porch"]

        #############################################################
        # run: 031 synthetic stuff                               ####
        #############################################################
        self.data.loc[:, "bed_to_bath_ratio"] = self.data.loc[:,"Bedroom AbvGr"] / ((self.data.loc[:,"total_baths"] + 1.0) * 1.0)
        # add np.exp(1) ie 'e' so that lot area is always at least 1.0
        self.data.loc[:, "log_lot_area"] = np.log(self.data.loc[:, "Lot Area"] + np.exp(1))
        self.data.pop("Lot Area")
        self.data.loc[:, "house_to_land_ratio"] = self.data.loc[:, "log_home_sf"] / self.data.loc[:,"log_lot_area"]

def strip_string(_str):
    '''
    strip parenthesis from strings and replace ' ' with '_'

    returns a string
    '''
    temp = str(_str)

    junk = "(){}"

    temp = "".join([c for c in temp if c not in junk])
    temp = temp.replace(" ", "_")

    return temp.lower()

def figure_months(years_since_sale, month_sold):
    '''
    Takes in how many years since sale and what month sold
    Figures out how long it has been since sale using assumption that last
    sale in data took place month 7 of 2010.  Therefore a house sold 7-2010
    would have a value of 0. A house sold on 6-2009 should have a value of 13

    Returns integer that reflects the number of months since a house was sold
    '''
    # variable with guardian variable to help indicate that something went wrong
    out_months = -1

    # handle case that the house was sold in the last year of the sample
    if years_since_sale == 0:
        # the latest date of sale from the sample is month 7 at years since sold 0
        out_months = 7 - month_sold

        return out_months

    else:
        # years_since_sale >= 1
        #
        # 6-2009 (1yr 6mo) => 6 + 7 => 13
        # 3-2007 (3yr 3mo) => 9 + 24 + 7 => 40
        out_months = (years_since_sale - 1) * 12 + (12 - month_sold) + 7

        return out_months

def simple_inflation_adjuster(mss, cpis):
    '''
    Looks up CPI multiplier sourced from the St Louis Federal Reserve
    mss: months since sold
    cpis: np array of adjustments sorted in reverse order so the higher
          the index the later the month is

    Returns CPI multiplier (float, 1.0 - ~1.10)
    '''

    return cpis[int(mss)]

def load_cpi_data():
    '''
    Load CPI data from the St Louis Federal Reserve
    # https://fred.stlouisfed.org/series/CPIAUCNS  (saved locally in this case)

    Returns numpy array of monthly inflation data with Jan, 2006 = 100
    '''
    cpi = pd.read_csv("../data/CPIAUCNS.csv")

    # 198.3 is the value of inflation multiplier as of Jan, 2006
    # dividing by this value to normalize the data for this sample to basis
    # of 100
    cpi.loc[:,"cpi"] = cpi.loc[:,"CPIAUCNS"] / 198.3

    # reverse the order of the np_cpi and then we won't have to do anything fancy
    # but just feed in the months since sold
    np_cpi = np.array(cpi.cpi.values)[::-1]
    # print np_cpi[50] # 1.02118003026  (seems about right)

    return np_cpi
