import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib
import seaborn as sns
import json
from collections import defaultdict


class ProcessedGameState:


  def __init__(self, file_path):
  	  '''
        initializing file path
      '''
      self.file_path = file_path


  def extract(self):
      '''
        function to extract data from parquet file
      '''

      df = pd.read_parquet(self.file_path, engine='pyarrow')
      return df
  

  def explore(self, data):
      '''
        understanding the- what columns it has, count of null values, does it have duplicate values(same location on same time would mean there's error in the data)
        understanding how many rounds were played
        checking all the area names on the map given
        checking names of players playing the game
        checking names of players in team 2
      '''

      print(data.info())
      print(data.duplicated(['round_num','tick','x','y','z']).sum())
      print(data.describe())
      print(data.head())
      print('\nTotal Number of rounds played:',data['round_num'].unique())
      print('\nArea names on the given map:',data['area_name'].unique())
      print('\nNames of players playing the game:',data['player'].unique())
      team2 = data[data['team']=='Team2']
      print('\nPlayers playing for team2:',team2['player'].unique())
      print('\n\n')

      x_avg_points = data.groupby('area_name')['x'].mean()
      y_avg_points = data.groupby('area_name')['y'].mean()

      fig, ax = plt.subplots(figsize=(10, 8)) 

      ax.scatter(data['x'], data['y'], color='green', label='all_points')
      ax.scatter(x_avg_points.values,y_avg_points.values, color='black', label='area_names')

      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_title('Plot of Whole Map with Area Names')
      ax.legend()
      fig.show()

      for key, value in x_avg_points.items():
        ax.text(x_avg_points[key], y_avg_points[key], key, va='bottom', ha='center')


  def rows_within_boundary(self,boundary_points,z,data):
      '''
        given boundary points - vertices of a polygon/boundary
        given constraints on z axis
        given data points
        stores status of a row in 'within_boundary' column where True means the row is in inside the boundary specified and False means it is not inside the specified boundary
      '''

      boundary_path = mpl_path.Path(boundary_points)
      data['coordinates'] = data.apply(lambda row: [row['x'], row['y']], axis=1)
      data['within_boundary'] = data['coordinates'].apply(lambda point: boundary_path.contains_point(point))
      print('\nCount of points within x,y axes boundary of vertices 13,14,15,16,17: ', (data['within_boundary'] == True).sum())
      data['within_boundary'] = np.where((data['z'] >= z[0]) & (data['z'] <= z[1]) & (data['within_boundary']==True), True, False)
      print('\nCount of points within the boundary after considering the Z axis: ',(data['within_boundary'] == True).sum())
      print('\nStatus of whether or not each row falls within the provided boundary is stored in "within_boundary" column.')

      return data


  def visualize_rows_within_boundary(self,x,y,x_filtered,y_filtered,x_outside_boundary,y_outside_boundary,x_outside_boundary_scaled,y_outside_boundary_scaled,data):
      '''
        using all the row's coordinates to make a pseudo map
        plotting the boundary on the map
        plotting the points within the specified boundary on map
      '''

      boundary_points = np.column_stack((x, y))
      boundary_path = mpl_path.Path(boundary_points)

      for index, value in data['within_boundary'].items():
        if value == True:
          x_filtered.append(data['x'][index])
          y_filtered.append(data['y'][index])
        else:
          if not boundary_path.contains_point(data['coordinates'][index]):
            x_outside_boundary.append(data['x'][index])
            y_outside_boundary.append(data['y'][index])
            if data['x'][index]>-3000 and data['y'][index]>-1000:
              x_outside_boundary_scaled.append(data['x'][index])
              y_outside_boundary_scaled.append(data['y'][index])

      fig, ax = plt.subplots(2,1,figsize=(10, 16)) 
      ax[0].scatter(x, y, color='blue', label='boundary_points')
      ax[0].scatter(x_filtered, y_filtered, color='red', label='within_boundary_points')
      ax[0].plot(x+x[:1], y+y[:1], color='black', label='boundary_line')

      ax[0].set_xlabel('X')
      ax[0].set_ylabel('Y')
      ax[0].set_title('Plot of Light Blue Area')
      ax[0].legend()

      print('\n\n')

      ax[1].scatter(x_outside_boundary, y_outside_boundary, color='green', label='outside_boundary_points')
      ax[1].scatter(x, y, color='blue', label='boundary_points')
      ax[1].scatter(x_filtered, y_filtered, color='red', label='within_boundary_points')
      ax[1].plot(x+x[:1], y+y[:1], color='black', label='boundary_line')

      ax[1].set_xlabel('X')
      ax[1].set_ylabel('Y')
      ax[1].set_title('Plot of Whole Map')
      ax[1].legend()

      plt.tight_layout()
      plt.show()

      return x_filtered,y_filtered,x_outside_boundary,y_outside_boundary,x_outside_boundary_scaled,y_outside_boundary_scaled


  def visualize_scaled_version(self,x,y,xnv,ynv,x_bsb,y_bsb,x_filtered,y_filtered,x_outside_boundary_scaled,y_outside_boundary_scaled,data):
      '''
        for the specific portion of map given in the assesment, showing a scaled version of the map with the larger boundary
      '''

      print('\n\n')
      fig, ax = plt.subplots(figsize=(10, 8)) 
      ax.scatter(x_outside_boundary_scaled, y_outside_boundary_scaled, color='green', label='outside_boundary_points')
      ax.scatter(x, y, color='blue', label='boundary_points')
      ax.scatter(x_filtered, y_filtered, color='red', label='within_boundary_points')
      ax.plot(x+x[:1], y+y[:1], color='blue', label='boundary_line')
      ax.scatter(xnv,ynv, color='blue')
      ax.plot(xnv, ynv, color='blue')

      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_title('Plot of Scaled Map')
      ax.legend()

      for key, value in x_bsb.items():
        ax.text(x_bsb[key], y_bsb[key]-25, key, va='top', ha='center',fontsize=11)

      plt.show()

      
  def get_weapons_class(self,data):
      '''
        extracting weapon_class from 'inventory' column and storing all the weapons in 'weapon' column
      '''

      weapons = []
      weapons_kind=defaultdict(lambda: 0)
      for value in data['inventory']:
        if value is not None:
          temp = []
          for each_weapon in value:
            if each_weapon!=None and 'weapon_class' in each_weapon:
                temp.append(each_weapon['weapon_class'])
                weapons_kind[each_weapon['weapon_class']]+=1
          weapons.append(temp)
        else:
          weapons.append(None)

      data['weapons']=weapons
      print('\n\nExtracted the weapon classes from the inventory and stored in "weapons" column.')
      print('\n')

      weapons_x = list(weapons_kind.keys())
      weapons_y = list(weapons_kind.values())

      plt.bar(weapons_x, weapons_y)
      plt.xlabel('Type')
      plt.ylabel('Count of Rows')
      plt.title('Count of Weapons Encountered Plot')
      plt.show()

      return data


  def get_team2_T_data(self,data):
      '''
        retrieving smaller subset of the dataset where the rows are of Team 2 playing on T(terrorist) side
      '''

      condition1 = data['is_alive'] == True
      condition2 = data['team'] == 'Team2'
      condition3 = data['side'] == 'T'

      T_result = data[condition1 & condition2 & condition3]

      return T_result

  def count_boundary_crossings(self,data,t_data,boundary_points_BombsiteB,x,y,xnv,ynv):
      '''
        for every round team 2 plays as T
        for every player of team 2
        checked if currently the player is in light blue area then on the next tick are they entering the dark blue boundary
        checked if they enter the bigger dark blue boundary coordinates via any other area
        returned both the counts
      '''

      count_light_blue = 0
      count_other_areas = 0

      boundary_path_dark_blue = mpl_path.Path(boundary_points_BombsiteB)

      for round in t_data['round_num'].unique():
        round_wise = t_data[t_data['round_num'] == round]
        
        for player in round_wise['player'].unique():
          player_wise = round_wise[round_wise['player']==player]

          sorted_time = player_wise.sort_values('tick')

          sorted_time['next_coordinate'] = sorted_time['coordinates'].shift(-1)
          sorted_time = sorted_time[:-1]
          sorted_time['next_tick'] = sorted_time['next_coordinate'].apply(lambda point: boundary_path_dark_blue.contains_point(point))
          
          self.plot_player_path(data,sorted_time,x,y,xnv,ynv,player,round)
          

          for index, row in sorted_time.iterrows():

            if row['within_boundary']==True and row['next_tick']==True:
              count_light_blue += 1
            
            elif boundary_path_dark_blue.contains_point(row['coordinates'])!=True and row['next_tick']==True:
              count_other_areas += 1

      return count_light_blue, count_other_areas


  def plot_player_path(self,data,sorted_time,x,y,xnv,ynv,player,round):
      '''
        plotting the trajectory of a player in a round of the game
      '''

      sorted_time['map_val'] = sorted_time['clock_time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

      fig, ax = plt.subplots(figsize=(10, 8)) 

      ax.scatter(data['x'], data['y'], color='black', label='all_points')
      cmap = matplotlib.colormaps['cool']
      scatter=ax.scatter(sorted_time['x'], sorted_time['y'], c=sorted_time['map_val'], cmap=cmap)
      ax.plot(x+x[:1], y+y[:1], color='blue', label='boundary_line')
      ax.plot(xnv, ynv, color='blue')

      cbar = plt.colorbar(scatter)
      cbar.set_label('Clock Timer in Seconds')
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      map_title = 'Path of '+player+' in round '+str(round) +' on Map'
      ax.set_title(map_title)
      ax.legend()
      fig.show()


  def calculate_avg_clock_time(self,t_data):
      '''
        calculating the average clock timer for all the rounds team2 plays as T
        calculating the overall average of all rounds combined
      '''

      counters = []

      for round in t_data['round_num'].unique():
          round_wise = t_data[t_data['round_num'] == round]
          round_counter = self.calculate_round_avg_clock_time(round_wise)
          if round_counter!=0:
            counters.append(round_counter)

      overall_avg_clock_time = (sum(counters) / len(counters)) / 60

      return counters, overall_avg_clock_time


  def calculate_round_avg_clock_time(self,round_data):
      '''
        for a particular round
        for every player, sorted their data according to tick
        stored their next encountered location in 'next_area' column
        check if the next area they enter is 'BombsiteB' stored the value in terms of True or False in 'target_location'
        keeping track of every player entering and exiting BombsiteB and updating the ammo of that team in BombsiteB
        returning the average clock timer for this particular round
      '''

      combined_data = pd.DataFrame()

      for player in round_data['player'].unique():
          player_wise = round_data[round_data['player'] == player]
          sorted_time = player_wise.sort_values('tick')
          sorted_time['next_area'] = sorted_time['area_name'].shift(-1)
          sorted_time = sorted_time[:-1]
          sorted_time['target_location'] = sorted_time['next_area'] == 'BombsiteB'
          combined_data = pd.concat([combined_data, sorted_time])

      round_counter = 0
      people_counter = 0
      weapon_count = {'Rifle': 0, 'SMG': 0}

      for index, row in combined_data.iterrows():
          if row['area_name'] != 'BombsiteB' and row['target_location'] == True:
              if 'SMG' in row['weapons']:
                  weapon_count['SMG'] += 1
              if 'Rifle' in row['weapons']:
                  weapon_count['Rifle'] += 1
              if weapon_count['Rifle'] >= 2 or weapon_count['SMG'] >= 2:
                  people_counter += 1
                  round_counter += int(row['clock_time'].split(':')[0]) * 60 + int(row['clock_time'].split(':')[1])

          elif row['area_name'] == 'BombsiteB' and row['target_location'] != True:
              if 'SMG' in row['weapons'] and weapon_count['SMG'] > 0:
                  weapon_count['SMG'] -= 1
              if 'Rifle' in row['weapons'] and weapon_count['Rifle'] > 0:
                  weapon_count['Rifle'] -= 1

      if people_counter != 0:
          round_avg_clock_time = round_counter / people_counter
          return round_avg_clock_time
      else:
          return 0


  def get_team2_CT_BombsiteB_data(self,data):
    '''
        retrieving smaller subset of the dataset where the rows are of Team 2 playing on CT(terrorist) side in 'BombsiteB' area
    '''

    condition1 = data['is_alive'] == True
    condition2 = data['team'] == 'Team2'
    condition3 = data['side'] == 'CT'
    condition4 = data['area_name'] == 'BombsiteB'

    CT_result = data[condition1 & condition2 & condition3 & condition4]

    return CT_result

  
  def generate_heatmap(self,ct_data):
    '''
      heatmap to see frequency of players on various bins of coordinates
    '''

    fig, ax = plt.subplots(figsize=(10, 8))

    heatmap, xedges, yedges = np.histogram2d(ct_data['x'], ct_data['y'], bins=20)

    plt.imshow(heatmap.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')
    plt.colorbar(label='Count')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heatmap from Coordinates')
    plt.show()


  def generate_overlayed_scatter_plot(self,data,ct_data):
  	  '''
        scatter plot of the required players on the map overlapping all other player positions
      '''

      fig, ax = plt.subplots(figsize=(10, 8))
      ax.scatter(data['x'], data['y'], color='black', label='all_points')
      ax.scatter(ct_data['x'], ct_data['y'], color='red', label='Team2 CT in BombsiteB')
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_title('Plot of Team2 CT players in BombsiteB overlapping all player points on the Whole Map')
      ax.legend()
      plt.show()


  def generate_overlayed_hexbin_plot(self,data,ct_data):
      '''
        function to overlap a heatmap structure on the map to better understand the position of players on the map
      '''

      fig, ax = plt.subplots(figsize=(10, 8))
      ax.scatter(data['x'], data['y'], color='black', label='all_points')
      plt.hexbin(ct_data['x'], ct_data['y'], gridsize=7, cmap='cool', alpha=0.8)
      plt.colorbar(label='Frequency Team2 CT in BombsiteB')
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_title('Plot of Frequency Team2 CT players in BombsiteB overlapping all player points on the Whole Map')
      ax.legend()
      plt.show()


  def load(self,data):
      '''
        loading the transformed data into a new paraquet file
      '''
      data.to_parquet('transformed_game_state_frame_data.parquet')


# coordinates of the light blue boundary are given

x = [-1735, -2024, -2806, -2472, -1565]
y = [250, 398, 742, 1233, 580]
z = [285,421]

x_filtered = []
y_filtered = []
x_outside_boundary = []
y_outside_boundary = []
x_outside_boundary_scaled = []
y_outside_boundary_scaled = []


# estimated the coordinates of dark blue boundary using cv and scaling them

x_bsb={
    '1':-1420,
    '2':-100,
    '3':-100,
    '4':-770,
    '5':-770,
    '6':-1065,
    '7':-1065,
    '8':-1420,
    '9':-1735,
    '10':-1735,
    '11':-2100,
    '12':-2100,
    '14':-2024,
    '13':-1735,
    '17':-1565,
}

y_bsb={
    '1':1000,
    '2':1000,
    '3':-550,
    '4':-550,
    '5':-450,
    '6':-450,
    '7':-330,
    '8':-300,
    '9':-430,
    '10':-650,
    '11':-650,
    '12':133,
    '14':398,
    '13':250,
    '17':580,
}


boundary_points = np.array([x, y])
boundary_points = boundary_points.transpose()

xnv = [value for value in x_bsb.values()]
ynv =  [value for value in y_bsb.values()]

boundary_points_BombsiteB = np.array([xnv,ynv])
boundary_points_BombsiteB = boundary_points_BombsiteB.transpose()

xnv.append(xnv[0])
ynv.append(ynv[0])

parquet_file = 'game_state_frame_data.parquet' 

game_data = ProcessedGameState(parquet_file) #q1.a

data = game_data.extract() #q1.a

game_data.explore(data) #q1.a

data = game_data.rows_within_boundary(boundary_points,z,data) #q1.b

x_filtered,y_filtered,x_outside_boundary,y_outside_boundary,x_outside_boundary_scaled,y_outside_boundary_scaled = game_data.visualize_rows_within_boundary(x,y,x_filtered,y_filtered,x_outside_boundary,y_outside_boundary,x_outside_boundary_scaled,y_outside_boundary_scaled,data) #q1.b

game_data.visualize_scaled_version(x,y,xnv,ynv,x_bsb,y_bsb,x_filtered,y_filtered,x_outside_boundary_scaled,y_outside_boundary_scaled,data) #q1.b

data=game_data.get_weapons_class(data) #q1.c

game_data.load(data) #q1.a

#q2.a 

t_data = game_data.get_team2_T_data(data)

print('Number of Team 2 T side inside the light blue boundary:',len(t_data[t_data['within_boundary']==True]))

count_light_blue, count_other_areas = game_data.count_boundary_crossings(data,t_data,boundary_points_BombsiteB,x,y,xnv,ynv)

print('Number of times a player from team2 T side enters the dark blue boundary via light blue boundary area',count_light_blue) # No, not a common strategy
print('Number of times a player from team2 T side enters the dark blue boundary via any other area except the light blue boundary area: ',count_other_areas)

#q2.b

counters, overall_avg_clock_time = game_data.calculate_avg_clock_time(t_data)

print('Average clock timer in seconds across different rounds:', counters)
print('Overall average timer in minutes that Team2 on T (terrorist) side enters “BombsiteB” with least 2 rifles or SMGs:',overall_avg_clock_time)

#q2.c

ct_data = game_data.get_team2_CT_BombsiteB_data(data)
game_data.generate_heatmap(ct_data)
game_data.generate_overlayed_scatter_plot(data,ct_data)
game_data.generate_overlayed_hexbin_plot(data,ct_data)