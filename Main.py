import os
import argparse
import zipfile
import io
import pandas as pd
import numpy as np
import contextlib
import json
from matplotlib import pyplot as plt
from matplotlib import rcParams


# retrieve argv elements and assign them to variables.
arg_parser = argparse.ArgumentParser(description="Supply arguments for -i, -o, -p, and -s")
arg_parser.add_argument('-i', '--input', help='Input zip directory (e.g. C:/Cap1_DC)', required=True)
arg_parser.add_argument('-o', '--output', help='Output json directory (e.g. C:/Cap1_Output)', required=True)
arg_parser.add_argument('-s', '--states', help="string of states (e.g. 'VA' 'DE' 'WV')",
                        required=False, nargs='*')
arg_parser.add_argument('-f', '--filter', help='Conventional Conforming Filter Flag True/False',
                        required=False)
arg_parser.add_argument('-p', '--plots', help='True/False for running plots of data', required=True)
args = arg_parser.parse_args()

# Example usage: >python Main.py -i 'C:/Cap1_DC' -o 'C:/Cap1_Output' -s 'VA', 'DE', 'WV' -f True -p True
# Set the paths to the data sources
raw_data_path = args.input
dir_loc = 'prod/user/sam/coaf/adhoc/tjy118/data/HMDA_Data/'
inst_zip = os.path.join(raw_data_path, '2012_to_2014_institutions_data.zip')
inst_file = dir_loc + '2012_to_2014_institutions_data.csv'
loans_zip = os.path.join(raw_data_path, '2012_to_2014_loans_data.zip')
loans_file = dir_loc + '2012_to_2014_loans_data.csv'
output_dir = args.output
plotting = args.plots
if args.states:
    states_filter = args.states
else:
    states_filter = None
if args.filter:
    con_filter = args.filter
else:
    con_filter = False


class FileBuilder(object):
    def __init__(self, in_zpath, in_fpath, loan_zpath, loan_fpath):
        self.in_zpath = in_zpath
        self.in_fpath = in_fpath
        self.loan_zpath = loan_zpath
        self.loan_fpath = loan_fpath

    def zip_reader(self, zpath, fpath):
        """Method for reading the contents out of the zip archive and returning the contents as a data frame"""
        with zipfile.ZipFile(zpath) as z_directory:
            # convert the zip directory to a seekable in-memory file type
            zipped_data = io.BytesIO(z_directory.read(fpath))
            # utilize low_memory=False due to mixed data types in some fields.
            # return the data frame from the zipped .csv file
            return pd.read_csv(zipped_data, low_memory=False, sep=',', header=0)

    def file_builder(self):
        """Method for returning the two files utilizing the zip_reader method"""
        ins_data = self.zip_reader(self.in_zpath, self.in_fpath)
        ln_data = self.zip_reader(self.loan_zpath, self.loan_fpath)
        return ins_data, ln_data

    def raw_file_join(self):
        ins_data, ln_data = self.file_builder()
        return pd.merge(ins_data, ln_data, how='Left', on=[
            'As_of_Year', 'Respondent_ID', 'Agency_Code'])


def zip_code_fix(data_file, zip_field):
    """Function for formatting the zip codes to standard zip_5"""
    data_file[zip_field] = data_file[zip_field].astype(str)
    # Use pandas built in row-wise iterator to strip the excess ZIP+4 data
    data_file[zip_field] = data_file[zip_field].str[:5]
    # Many of the New England zip codes that are prefixed with '0' have stripped data
    # fix this with a numpy where conditional to find length 4 and
    data_file[zip_field] = np.where(data_file[zip_field].str.len() == 4,
                                    '0' + data_file[zip_field], data_file[zip_field])
    # Puerto Rico zip codes are formatted incorrectly.  fix in a similar way,
    # but since NaN is now a string, make sure to not modify it with a string length
    # conditional check.
    data_file[zip_field] = np.where((data_file[zip_field].str.len() == 3) &
                                    (data_file[zip_field] != 'nan'),
                                    '00' + data_file[zip_field], data_file[zip_field])
    return data_file


def convert_to_num(data_file, field):
    """Function to change data types to numbers as needed"""
    data_file[field] = data_file[field].astype(str)
    # strip out the Excel-type 'NA   ' white space
    data_file[field] = data_file[field].map(str.strip)
    # Convert the 'NA' value to a null value
    data_file[field] = np.where(data_file[field] == 'NA', None, data_file[field])
    # convert the field to floats, but if any value is invalid, return NaN in its failure place
    data_file[field] = pd.to_numeric(data_file[field], errors='coerce')
    return data_file


def lookup_create(field_list, source_frame):
    """Function for returning a lookup reference table based on the dataframe"""
    return source_frame[field_list].drop_duplicates().reset_index().drop('index', axis=1)


def state_convert(states):
    """function to convert a string input to a list"""
    if type(states) == str:
        state_list = [s.strip().upper() for s in states.split(',')]
    else:
        state_list = [s.upper() for s in states]
    return state_list


def state_verify(states, source_data):
    """Function for verifying the accuracy / format user input state list or string"""
    # Get the list of unique states present in the loan data
    states_raw = lookup_create(['State'], source_data)
    # Convert the dataframe of States to a list
    state_full = states_raw['State'].tolist()
    # state_list will be the returned variable for this function
    state_list = []
    invalid_list = []
    # if the default value is used in hmda_to_json method (None) then return full list
    if states is None:
        state_list = state_full
        print('Evaluating for state(s): %r' % state_list)
    # check to make sure that the states provided by the user are available in the loan data
    else:
        input_states = state_convert(states)
        for elem in input_states:
            if elem in state_full:
                state_list.append(elem)
            else:
                invalid_list.append(elem)
        # Report which entries are invalid.
        if invalid_list:
            print('The following state(s) are invalid: %r' % invalid_list)
        print('Evaluating for state(s): %r' % state_list)
    return state_list


def conforming_filter(data):
    """Function for returning a subset of the dataframe data that is only Conventional Conforming"""
    data = data.loc[data['Conventional_Conforming_Flag'] == 'Y']
    return data


def directory_check_create(directory):
    """Function for checking if the save directory exists, and if not, to create it"""
    if not os.path.isdir(directory):
        os.mkdir(directory)


def cleanup_old(filepath):
    """Function for deleting any previous runs of the code's json output."""
    # don't throw an exception if the file isn't found, otherwise, remove the file.
    with contextlib.suppress(FileNotFoundError):
        os.remove(filepath)


def market_size(full_data, output_path, conforming_check):
    """Function for plotting market size data for top lenders in each state"""
    # define the save path for the charts
    save_path_dir = output_path + '/Plots'
    directory_check_create(save_path_dir)
    state_lender_dir = save_path_dir + '/Top_Lenders'
    directory_check_create(state_lender_dir)
    # filter the conventional conforming data if passed in.
    if conforming_check:
        full_data = conforming_filter(full_data)
    # return a dataframe with the mortgages issued by Lender, state, and year.
    lender_group = pd.DataFrame({'Mortgage_Count': full_data.groupby(['As_of_Year', 'Respondent_Name_TS',
                                                                      'State']).size()}).reset_index()
    # output this file as a json file to show which lenders do the most business in each state.
    lender_json = lender_group.reset_index(drop=True).to_json(orient='records')
    json_dir = output_path + '/Market_Share'
    directory_check_create(json_dir)
    cleanup_old(json_dir + 'Lender_Market_Share.json')
    market_write = open(json_dir + '/Lender_Market_Share.json', 'w')
    # remove the previous run's files if they exist
    file_listing = [file for file in os.listdir(state_lender_dir) if file.endswith(".png")]
    for file in file_listing:
        os.remove(state_lender_dir + '/' + file)
    # write the json stream to the output file
    json.dump(lender_json, market_write)
    market_write.close()
    # report out the largest lenders in each state
    state_list = lookup_create('State', lender_group)
    for state in state_list['State']:
        lender_state = lender_group.loc[lender_group['State'] == state]
        # sort the list by the most recent year, then by size of lender accounts.
        lender_state = lender_state.sort_values(['As_of_Year', 'Mortgage_Count'], ascending=[False, False])
        # get the top 10 lenders for the most recent year
        top_lenders = lender_state['Respondent_Name_TS'][:10]
        # get the full history of the top 10 lenders in the most recent year
        top_lender_data = lender_state[lender_state['Respondent_Name_TS'].isin(top_lenders)].reset_index()
        # pivot the table to arrange the data for plotting in a stacked bar chart
        top_lender_grouped = top_lender_data.pivot(index='Respondent_Name_TS', columns='As_of_Year',
                                                   values='Mortgage_Count').reset_index()
        rcParams['figure.figsize'] = 10, 10
        n = 10
        # acquire the array for each year
        data1 = np.array(top_lender_grouped[2012])
        data2 = np.array(top_lender_grouped[2013])
        data3 = np.array(top_lender_grouped[2014])
        # get the attributes of the subplots
        fig, ax = plt.subplots()
        # set the x axis sequence (0, 1, 2, 3...9)
        bar_locations = np.arange(n)
        # add the data to the chart
        ax.bar(bar_locations, data1, label='2012')
        # add the next year's data to the chart, having it's bottom be the top of the previous data's bar.
        ax.bar(bar_locations, data2, bottom=data1, color='r', label='2013')
        # add the final year's data, bottom location being set by the sum of the array positions for the
        # preceding data sets.
        ax.bar(bar_locations, data3, bottom=np.array(data2) + np.array(data1), color='g', label='2014')
        plt.xlabel(top_lender_grouped['Respondent_Name_TS'])
        plt.legend(loc='upper left')
        plt.title('Number of Mortgages in %s issued by Top Regional Lenders' % state)
        # adjust the plot size in order to see the legend for the x axis
        fig.subplots_adjust(bottom=0.28)
        plt.xticks(range(10), range(10))
        plt.savefig(state_lender_dir + '/%s.png' % state)
        plt.close()


def county_income_plot(full_data, save_path, conforming_check):
    """Function for plotting a distribution by county of Applicant Income"""
    state_listing = lookup_create('State', full_data)
    # create the directories for the plots
    save_path_plots = save_path + '/Plots'
    directory_check_create(save_path_plots)
    save_path_state = save_path_plots + '/County_Plots'
    directory_check_create(save_path_state)
    # loop through each state
    for state in state_listing['State']:
        # cleanup / create the state directory for county data
        state_dir = save_path_state + '/' + state
        directory_check_create(state_dir)
        # create a subset dataframe by state
        state_file = loan_data.loc[loan_data['State'] == state]
        # filter conventional_conforming if requested by sys.argv
        if conforming_check:
            state_file = conforming_filter(state_file)
        # filter out extreme outliers in order to see the majority of the market.
        state_file = state_file.loc[state_file['Applicant_Income_000'] < 250]
        # remove missing County_Name and State data
        state_file = state_file.dropna(how='any', subset=['County_Name', 'State'])
        # fill in missing Applicant_Income_000 data with the column median value
        state_file['Applicant_Income_000'] = \
            state_file['Applicant_Income_000'].fillna(state_file['Applicant_Income_000'].median())
        # get a list of the counties in the state
        county_listing = lookup_create('County_Name', state_file)
        # cleanup the old files if they've already been created
        file_listing = [file for file in os.listdir(state_dir) if file.endswith(".png")]
        for file in file_listing:
            os.remove(state_dir + '/' + file)
        # create a distribution plot (histogram) for each county and save it in the state directory
        for county in county_listing['County_Name']:
            county_file = state_file.loc[state_file['County_Name'] == county]
            rcParams['figure.figsize'] = 15, 10
            bins = 50
            county_file.hist(column='Applicant_Income_000', bins=bins, range=(0, 250))
            plt.title('Distribution of Income in %s, %s' % (county, state))
            plt.xlabel('Total Household Income')
            plt.ylabel('Number of households in income band')
            plt.savefig(state_dir + '/%s.png' % county, bbox_inches='tight')
            plt.close()


def total_market(full_data, output_path, conforming_check):
    """Function for plotting the total market size for each state by year"""
    # define the save directory, create it if it doesn't exist.
    save_path_dir = output_path + '/Plots'
    directory_check_create(save_path_dir)
    # this plot will be in a subfolder.  Create it if it doesn't exist.
    market_dir = save_path_dir + '/Total_Market'
    directory_check_create(market_dir)
    # Clear out the destination directory for subsequent runs.
    file_listing = [file for file in os.listdir(market_dir) if file.endswith(".png")]
    for file in file_listing:
        os.remove(market_dir + '/' + file)
    # if only conventional_conforming data is desired, return the data set with only that info.
    if conforming_check:
        full_data = conforming_filter(full_data)
    # group the data set to get a count of mortgages issued by state and year.
    state_group = pd.DataFrame({'Mortgages': full_data.groupby(['As_of_Year', 'State']).size()
                                }).reset_index()
    # convert the data type to a datetime element for charting purposes.
    state_group['As_of_Year'] = pd.to_datetime(state_group['As_of_Year'].astype(str))
    # extract the information for each state for each year
    state_list = lookup_create('State', state_group)
    # plot each state's data on the figure by looping through the state listing.
    for state in state_list['State']:
        state_vals = state_group.loc[state_group['State'] == state]
        plt.plot(state_vals['As_of_Year'], state_vals['Mortgages'], label=state, linewidth=5)
    plt.xticks(rotation=45)
    plt.ylabel('Count of Mortgages issued within each state')
    plt.xlabel('Year')
    plt.legend(loc='upper right')
    # save the plot, then close it.
    plt.savefig(market_dir + '/Total_Market.png', bbox_inches='tight')
    plt.close()


class HMDA(object):
    """Main Class for generating file / formatting / saving JSON data from zipped .csv files"""
    def __init__(self, institution_zip_file, institution_csv_file, loan_zip_file, loans_csv_file):
        self.inst_fp = institution_zip_file
        self.inst_file = institution_csv_file
        self.loans_fp = loan_zip_file
        self.loans_file = loans_csv_file
        self.ins_data = None
        self.ln_data = None
        self.resp_ref = None
        self.full_file = None
        self.state_list = None

    def hmda_init(self):
        """Method for reading the data in and returning a full joined dataframe"""
        # Read in the data from the CSV zipped archive and build two dataframes of the raw data
        self.ins_data, self.ln_data = FileBuilder(self.inst_fp, self.inst_file,
                                                  self.loans_fp, self.loans_file).file_builder()
        # Adjust the zip code fields in the institution data for regional clustering purposes
        self.ins_data = zip_code_fix(self.ins_data, 'Respondent_ZIP_Code')
        self.ins_data = zip_code_fix(self.ins_data, 'Parent_ZIP_Code')
        # Convert fields to floats that are currently strings based on the source file formatting
        # and invalid NA signatures
        self.ln_data = convert_to_num(self.ln_data, 'Applicant_Income_000')
        self.ln_data = convert_to_num(self.ln_data, 'FFIEC_Median_Family_Income')
        self.ln_data = convert_to_num(self.ln_data, 'Number_of_Owner_Occupied_Units')
        self.ln_data = convert_to_num(self.ln_data, 'Tract_to_MSA_MD_Income_Pct')
        # Build the reference table:
        self.resp_ref = lookup_create(['Respondent_ID', 'Respondent_Name_TS', 'As_of_Year',
                                       'Respondent_City_TS', 'Respondent_State_TS', 'Respondent_ZIP_Code',
                                       'Parent_Name_TS', 'Parent_City_TS', 'Parent_State_TS',
                                       'Parent_ZIP_Code'], self.ins_data)
        # Merge the Loan Data to the Institution Data
        self.full_file = pd.merge(self.ln_data, self.resp_ref, how='inner', on=(
            'Respondent_ID', 'As_of_Year'))

        return self.full_file

    def hmda_to_json(self, data, dest_dir, states=None, conventional_conforming=False):
        """Method for the output of the data to json by state"""
        # get the state data
        # check if the directory exists.  If not, create it.
        directory_check_create(dest_dir)
        self.state_list = state_verify(states, data)
        for state in self.state_list:
            # create a new copy of the data frame for each state
            state_file = data.loc[data['State'] == state]
            # if the arg is supplied to provide only the conventional_conforming data then filter.
            if conventional_conforming:
                state_file = conforming_filter(state_file)
            # transform the dataframe into a json object
            state_output = state_file.reset_index(drop=True).to_json(orient='records')
            # check to see if state directories exist, and if not, create them
            directory_check_create(dest_dir + '/' + state)
            cleanup_old(dest_dir + '/' + state + '/' + state + '.json')
            state_write = open(dest_dir + '/' + state + '/' + state + '.json', 'w')
            json.dump(state_output, state_write)
            state_write.close()
        print('Files created in %s for the states: %r' % (dest_dir, self.state_list))

    def run_plots(self, data, dest_dir, c_filter=False):
        """Method for running the plots"""
        county_income_plot(data, dest_dir, c_filter)
        market_size(data, dest_dir, c_filter)
        total_market(data, dest_dir, c_filter)

# Run the program
if __name__ == '__main__':
    # Instantiate the HMDA class
    loans = HMDA(inst_zip, inst_file, loans_zip, loans_file)
    # Import the data
    loan_data = loans.hmda_init()
    # Run plotting if -p argument is True
    if plotting:
        loans.run_plots(loan_data, output_dir, con_filter)
    # output the zipped source as json files by state
    loans.hmda_to_json(loan_data, output_dir, states_filter, con_filter)


