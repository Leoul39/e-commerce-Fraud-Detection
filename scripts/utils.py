import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import seaborn as sns
log_dir = os.path.join(os.getcwd(), 'logs')

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
log_file_info = os.path.join(log_dir, 'Info.log')
log_file_error = os.path.join(log_dir, 'Error.log')

formatter = logging.Formatter('%(asctime)s - %(levelname)s :: %(message)s',
                                    datefmt='%Y-%m-%d %H:%M')

info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(info_handler)
logger.addHandler(error_handler)
def plot_data(data,column):
        """
        This function plots the mentioned column extracted out of the data. This function plots a 
        bar chart for catagorical columns and a histogram for numerical columns.
        Parameter:
           column: this can be a catagorical or numerical column
        Returns: 
            A bar chart for Catagorical Column and a histogram for numerical column
        """
        first = data[column].iloc[0]
        try:
            if isinstance(first,(np.int64,np.float64)):
                logger.info(f"Plotting histogram for {column} column")
                plt.hist(data[column],bins=100,color='steelblue', edgecolor='black')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.title(f'The histogram plot for {column} column')
                plt.show()
            elif isinstance(first,(object,bool)):
                logger.info(f"Plotting a bar chart for {column} column")
                d=data[column].value_counts()
                d.plot(kind='bar',color='steelblue', edgecolor='black', linewidth=1.5)
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.title(f"The distribution of values in {column} column")
                plt.show()
        except Exception as e:
             logger.error(f"An error has occured")
             print(f"Error as {e}")

def country(x,ip_df):
    """
    This function brings out the country by just entering the lower_bound ip adress and the upper_bound 
    ip address. This works for the ip address dataset.

    parameter:
       x- the row used
       ip_df- the ip address dataset
    returns:
       the country from those described ip addresses
    """
    try:
        country= ip_df.loc[((x >ip_df['lower_bound_ip_address']) & (x< ip_df['upper_bound_ip_address'])),'country']
        if not country.empty:
            return country.iloc[0]
        return None
    except Exception as e:
         logger.error(f"An error occured as {e}")
         return None
def bivariant_plot(data,columns):
    try:
        logger.info(f"Starting the barchar for {columns}")
        fig,ax=plt.subplots(nrows=1,ncols=len(columns), figsize=(18,8))
        for i,col in enumerate(columns):
            sns.countplot(x=col, data=data, hue='class', palette = 'deep', ax=ax[i])
            ax[i].set_title(f'Bar Chart for {col}')
            plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"An error has occured: {e}")
        return None
def geographical_plot(data1,data2):
    """
    This function lays the world map and shows the amount of fraudulent activity by countries, differentiating
    them with intensity.

    Parameter:
        data1- fraud dataset
        data2- world dataset
    Returns:
        geographical plot showing fraudulent activities
    """
    try:
        data1['country']=data1['country'].replace({'United States':'United States of America'})
        fraud_by_country= data1[data1['class']==1].groupby('country').size().reset_index(name='count')
        world_plot=data2.merge(fraud_by_country,left_on='ADMIN',right_on='country',how='left')
        world_plot.fillna(0,inplace=True)
        norm = cl.SymLogNorm(linthresh=500, linscale=1.0, vmin=50, vmax=6000)
        logger.info(f'Plotting the geographical plot for fraudulent activities')
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        world_plot.boundary.plot(ax=ax, linewidth=1)
        world_plot.plot(column='count', ax=ax, legend=True,
                legend_kwds={'label': "Count by Country",
                                'orientation': "horizontal"},
                cmap='Greens',norm=norm
                )
        plt.title("Country Counts Heatmap")
        plt.show()
    except Exception as e:
        logger.error(f"An error has occcured: {e}")
def feature_engineering(fraud_df):
    """
    This function generates many features from already existing features 

    Parameter:
        fraud_df- the fraud dataset
    Returns:
        a new dataframe with all the generated features included
    """
    try:
        logger.info('Generating many essential features from already existing features')
        fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
        fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])

        # Calculate transaction frequency
        transaction_frequency = fraud_df.groupby('device_id').size().reset_index(name='transaction_frequency')

        # Calculate total purchase value for each device
        total_purchase_value = fraud_df.groupby('device_id')['purchase_value'].sum().reset_index(name='total_purchase_value')

        # Merge frequency and total value
        user_metrics = transaction_frequency.merge(total_purchase_value, on='device_id',how='inner')

        # Calculate transaction velocity
        user_metrics['transaction_velocity'] = user_metrics['total_purchase_value'] / user_metrics['transaction_frequency']

        fraud_df=fraud_df.merge(user_metrics,on='device_id',how='inner')
        #Day of Week
        fraud_df['dayofweek']=fraud_df['purchase_time'].dt.dayofweek
        #Month
        fraud_df['month']=fraud_df['purchase_time'].dt.month
        #Day
        fraud_df['day']=fraud_df['purchase_time'].dt.day
        #Hour
        fraud_df['hour']=fraud_df['purchase_time'].dt.hour

        return fraud_df
    except Exception as e:
        logger.error(f"An error has occured: {e}")
        return None