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
                log_price=True):
        # load in data from csv, when exported from XLS used '~' as the
        # separator
        self.raw_data = pd.read_csv(housing_path, "~")

        # if this is true then adjust sale prices for inflation
        self.adjust_inflation = adjust_inflation

        # using log price?
        self.log_price = log_price

    def fit(self):
        # call method to do feature transformations, feature engineering
        self.manipulate_data()

        ##############################
        ### Y Label Manipulations ####
        ##############################

        # if we're using log price set Y to that and drop the normal 'SalePrice'
        if self.log_price:
            # if adjusting for inflation is true
            if self.adjust_inflation:
                # look up how many months since each prop was sold and adjust
                # the sale price to 2010 dollars
                self.raw_data.loc[:,"real_saleprice"] = self.raw_data.loc[:,"SalePrice"] * self.raw_data.loc[:,"cpi_adjustment"]
                self.raw_data.loc[:,"log_real_saleprice"] = np.log(self.raw_data.loc[:,"real_saleprice"])
                self.y = self.raw_data.loc[:,"log_real_saleprice"]

            # otherwise don't adjust log price for inflation
            else:
                self.raw_data.loc[:,"log_saleprice"] = np.log(self.raw_data.loc[:,"SalePrice"])
                self.y = self.raw_data.loc[:,"log_saleprice"]

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

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        return self

    def transform(self):
        # transform to log prices
        # self.y = np.log(y)
        return self.X, self.y

    def fit_transform(self):
        self.fit()

        # print "len X", len(self.X)
        self.transform()

    def dummify(self, column, col_text):
        '''
        take in categorical column and convert it to flagged dummy columns
        '''
        boink = pd.get_dummies(self.raw_data.loc[:,column], dummy_na=True)
        keys = boink.keys()
        # make key into a string to handle numberical categories (int)
        new_keys = [col_text + str(key).lower() for key in keys]
        boink.columns = new_keys
        self.data = pd.concat((self.data, boink), axis=1)


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
        # run: 005 room ratio                               #########
        #############################################################
        self.data.loc[:,"bed_to_room_ratio"] = self.data.loc[:,"Bedroom AbvGr"] \
                                    / (self.data.loc[:,"TotRms AbvGrd"] * 1.0)

        #############################################################
        # run: 006 yrs_since_update                         #########
        #############################################################
        self.data.loc[:,"yrs_since_update"] = 2010 -self.raw_data["Year Remod/Add"]

        #############################################################
        # run: 006 overall cond flags                       #########
        #############################################################
        self.dummify("Overall Cond", "oc_")

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

        #############################################################
        # run: 007 years since sold                         #########
        #############################################################

        # figure out how long since sale (in years)
        self.data.loc[:,"yrs_since_sold"] = 2010 - self.raw_data.loc[:,"Yr Sold"]

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

        # # bring months since sold to model
        # self.data.loc[:,"mon_since_sold"] = self.raw_data.loc[:, "mon_since_sold"]
        #
        # # bring in what month sold, also
        # self.data.loc[:, "month_sold"] = self.raw_data.loc[:,"Mo Sold"]
        #
        # #############################################################
        # # Zoning info                                       #########
        # #############################################################
        #
        # #raw_data.loc[:,"MS Zoning"].unique()
        # # array(['RL', 'RH', 'FV', 'RM', 'C (all)', 'I (all)', 'A (agr)'], dtype=object)
        # boink = pd.get_dummies(self.raw_data.loc[:,"MS Zoning"])
        #
        # boink.columns = ["zoning_rl", "zoning_rh", "zoning_fv", "zoning_rm", "zoning_c_all", "zoning_i_all", "zoning_a_all"]
        #
        # self.data = pd.concat((self.data, boink), axis=1)
        #
        # #############################################################
        # # Roof stuff                                        #########
        # #############################################################
        #
        # # self.raw_data.loc[:,"Roof Style"].unique()
        # # array(['Hip', 'Gable', 'Mansard', 'Gambrel', 'Shed', 'Flat'], dtype=object)
        #
        # boink = pd.get_dummies(self.raw_data.loc[:,"Roof Style"])
        # boink.columns = ["roof_hip", "roof_gable", "roof_mansard",
        #                 "roof_gambrel", "roof_shed", "roof_flat"]
        #
        # self.data = pd.concat((self.data, boink), axis=1)
        #
        # #############################################################
        # # Kitchen stuff                                     #########
        # #############################################################
        #
        # boink = pd.get_dummies(self.raw_data.loc[:,"Kitchen Qual"])
        # keys = boink.keys()
        #
        # # print boink.head()
        # # print keys
        # # Index([u'Ex', u'Fa', u'Gd', u'Po', u'TA'], dtype='object')
        # new_keys = ["kqual_" + key.lower() for key in keys]
        # boink.columns = new_keys
        # self.data = pd.concat((self.data, boink), axis=1)
        #
        # #############################################################
        # # Neighborhood flags                                #########
        # #############################################################
        # # print self.raw_data.loc[:,"Neighborhood"].unique()
        # boink = pd.get_dummies(self.raw_data.loc[:,"Neighborhood"])
        # keys = boink.keys()
        # new_keys = ["neigh_" + key.lower() for key in keys]
        # boink.columns = new_keys
        #
        # self.data = pd.concat((self.data, boink), axis=1)
        #
        # self.dummify("Lot Shape", "lot_shape_")
        #
        # #############################################################
        # # Condition types, road sizes near house?           #########
        # #############################################################
        #
        # self.dummify("Condition 1", "cond_1_")
        # self.dummify("Condition 2", "cond_2_")
        #
        # #############################################################
        # # fireplace stuff                                   #########
        # #############################################################
        #
        # self.dummify("Fireplace Qu", "fp_qual_")
        #
        # #############################################################
        # # sale type    and condition                        #########
        # #############################################################
        # self.dummify("Sale Condition", "sale_cond_")
        # self.dummify("Sale Type", "sale_type_")
        #
        # #############################################################
        # # house stype                                       #########
        # #############################################################
        # # self.dummify("House Style", "house_style_")
        #
        # self.dummify("Land Contour", "land_cont_")
        # self.dummify("Lot Config", "lot_conf_")
        # self.dummify("Land Slope", "land_slope_")
        # #############################################################
        # # exterior stuff                                    #########
        # #############################################################
        #
        # self.dummify("Exterior 1st", "ext_1_")
        # self.dummify("Exterior 2nd", "ext_2_")
        #
        # self.dummify("Exter Qual", "ext_qual_")
        # self.dummify("Exter Cond", "ext_cond_")
        #
        # self.dummify("Foundation", "foundation_")
        #
        # self.raw_data.loc[:, "Mas Vnr Type"].fillna("None", inplace=True)
        # self.dummify("Mas Vnr Type", "mas_vnr_typ_")
        # self.raw_data.loc[:,"Mas Vnr Area"].fillna(0, inplace=True)
        # self.data.loc[:,"Mas Vnr Area"] = self.raw_data.loc[:, "Mas Vnr Area"]
        #
        # #############################################################
        # # Much more basement stuff                          #########
        # #############################################################
        # # Actually I think pandas dummify will handle nan's ok
        # # self.raw_data.loc[:,"Bsmt Cond"].fillna("TA", inplace=True)
        # self.dummify("Bsmt Cond", "bsmt_cond_")
        #
        # self.dummify("Bsmt Exposure", "bsmt_expos_")
        #
        # self.dummify("BsmtFin Type 1", "bsmt_fin_typ1_")
        #
        # # seems like there is one nan value but can replace with 0
        # # probably the nan doesn't have a basement?
        # self.raw_data.loc[:, "BsmtFin SF 1"].fillna(0, inplace=True)
        # self.data.loc[:,"bsmtfin_sf_1"] = self.raw_data.loc[:, "BsmtFin SF 1"]
        #
        # self.dummify("BsmtFin Type 2", "bsmt_fin_typ2_")
        #
        # self.raw_data.loc[:,"BsmtFin SF 2"].fillna(0, inplace=True)
        # self.data.loc[:,"bsmtfin_sf_2"] = self.raw_data.loc[:, "BsmtFin SF 2"]
        #
        # self.raw_data.loc[:,"Bsmt Unf SF"].fillna(0, inplace=True)
        # self.data.loc[:,"bsmtunf_sf"] = self.raw_data.loc[:, "Bsmt Unf SF"]
        #
        # self.raw_data.loc[:,"Total Bsmt SF"].fillna(0, inplace=True)
        # self.data.loc[:,"Total Bsmt SF"] = self.raw_data.loc[:,"Total Bsmt SF"]

        #############################################################
        # Bldg type                                         #########
        #############################################################

        # self.dummify("Bldg Type", "bldg_type_")

        # print self.raw_data.loc[:,"Roof Matl"].unique()
        # ['CompShg' 'WdShake' 'Tar&Grv' 'WdShngl' 'Membran' 'ClyTile' 'Roll' 'Metal']
        # boink = pd.get_dummies(self.raw_data.loc[:, "Roof Matl"])
        # boink.columns = ["roof_mat_compshg", "roof_mat_wdshake",
        #         "roof_mat_targrv", "roof_mat_wdshngl", "roof_mat_membran",
        #         "roof_mat_clytile", "roof_mat_roll", "roof_mat_metal"]
        # self.data = pd.concat((self.data, boink), axis=1)

        #############################
        #############################
        ##### JUNKED FEATURES #######
        #############################
        #############################

        #
        # # amount of lot used ratio? lot / square_feet
        # # self.data.loc[:,"lot_home_area_ratio"] = self.raw_data.loc[:,"Lot Area"] \
        # #                         / (self.data.loc[:, "home_sf"] * 1.0)
        #
        # self.data.loc[:,"has_2nd_flr"] = self.raw_data.loc[:,"2nd Flr SF"] != 0
        #

        # self.data.loc[:,"Gr Liv Area"] = self.raw_data.loc[:, "Gr Liv Area"]

        # self.data.loc[:, "bed_to_bath_ratio"] = self.data.loc[:, "Bedroom AbvGr"] / (self.data.loc[:, "total_baths"] * 1.0)

        ##############################################################################
        # use the oneHot encoder to create dummy columns for overall quality #########
        ##############################################################################
        # encoder = OneHotEncoder()
        # temp = self.raw_data["Overall Qual"]
        # overall_qual_dummies = encoder.fit_transform(temp.values.reshape(-1, 1))
        # overall_qual_dummies = overall_qual_dummies.toarray()
        # print "overall_qual_dummies shape:", overall_qual_dummies.shape
        #
        # oq_dummies = pd.DataFrame(overall_qual_dummies, columns=["oq_1", "oq_2", "oq_3", "oq_4", "oq_5", "oq_6",
                                                                #  "oq_7", "oq_8", "oq_9", "oq_10"])
        # print oq_dummies.head()

        # self.data = pd.concat((self.data, oq_dummies), axis=1)
        #simple_data.pop("Overall Qual")

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
