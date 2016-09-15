import os
import sys
import argparse
import zipfile
import io
import pandas as pd
import numpy as np
import contextlib
import json

# retrieve argv elements and assign them to variables.
arg_parser = argparse.ArgumentParser(description="Supply arguments for -i, -o, and -s")
arg_parser.add_argument('-i', '--input', help='Input zip directory (e.g. C:/Cap1_DC)', required=True)
arg_parser.add_argument('-o', '--output', help='Output json directory (e.g. C:/Cap1_Output)', required=True)
arg_parser.add_argument('-s', '--states', help="string of states (e.g. 'VA' 'DE' 'WV')",
                        required=False, nargs='*')
arg_parser.add_argument('-f', '--filter', help='Conventional Conforming Filter Flag True/False',
                        required=False)
args = arg_parser.parse_args()

# Example usage: >python Main.py -i 'C:/Cap1_DC' -o 'C:/Cap1_Output' -s ['VA', 'DE'] -f True
# Set the paths to the data sources
raw_data_path = args.input
dir_loc = 'prod/user/sam/coaf/adhoc/tjy118/data/HMDA_Data/'
inst_zip = os.path.join(raw_data_path, '2012_to_2014_institutions_data.zip')
inst_file = dir_loc + '2012_to_2014_institutions_data.csv'
loans_zip = os.path.join(raw_data_path, '2012_to_2014_loans_data.zip')
loans_file = dir_loc + '2012_to_2014_loans_data.csv'
output_dir = args.output
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



# provide cleansing / outlier functions for key fields (loan amount / income / etc)

# provide charting capabilities.
# add some summary stats to the file to show std dev by state, by lender, type, etc.
# dedup the file by grouping on sequence_number and agency_code


# Test the HMDA class
test_r = HMDA(inst_zip, inst_file, loans_zip, loans_file)
test = test_r.hmda_init()

test_r.hmda_to_json(test, dest_dir=output_dir, states=states_filter, conventional_conforming=con_filter)






"""
# Firstly, we are only interested in the Conventional/Conforming loans.  Create this data frame.
cc_loans = loan_data[(loan_data['Conventional_Status'] == 'Conventional') & (
    loan_data['Conforming_Status'] == 'Conforming')]

# Verify that the Conventional/Conforming field is correct and fix it if it is not.
cc_verify = len(cc_loans[cc_loans['Conventional_Conforming_Flag'].str.contains('N')])
if cc_verify == 0:
    print('Number of incorrect entries in Conventional_Conforming_Flag : %d' % cc_verify)
else:
    print('Incorrect data present on %d rows. Resetting the data.' % cc_verify)
    # Run pandas indexer to search through all rows where the first statement is true and then change
    # those entries in that column to the correct value.
    cc_loans.ix[cc_loans.Conventional_Conforming_Flag == 'N', 'Conventional_Conforming_Flag'] = 'Y'
    cc_verify = len(cc_loans[cc_loans['Conventional_Conforming_Flag'].str.contains('N')])
    print('Incorrect entries fixed.  Invalid row count: %d' % cc_verify)
"""

"""
# create a lookup table from the loan data that represents the agency
agency_data = lookup_create(['Agency_Code', 'Agency_Code_Description'], loan_data)

# merge the loan data with the agency_lookup dataframe
institution_data_agency = pd.merge(institution_data, agency_data, how='left', on='Agency_Code')

# create a lookup table from the institutions data for respondent id to respondent name
respondent_data = lookup_create(['Respondent_ID', 'Respondent_Name_TS', 'Respondent_City_TS',
                                 'Respondent_State_TS', 'Respondent_ZIP_Code', 'As_of_Year'], institution_data)

# Assign a respondent name to each of the loan transactions.
loans_resp_data = pd.merge(cc_loans, respondent_data, how='left', on=('Respondent_ID', 'As_of_Year'))



ins_grp = pd.DataFrame({'count': institution_data.groupby(
    by=('Parent_Name_TS', 'Parent_City_TS', 'Parent_State_TS')).size()}).reset_index()
ins_grp_srt = ins_grp.sort_values(by='count', ascending=0)

"""
