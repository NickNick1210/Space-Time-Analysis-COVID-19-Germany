#!/usr/bin/env python
# coding: utf-8

# Jade Hochschule - Standort Oldenburg<br>
# Fachbereich: Bauwesen Geoinformation Gesundheitstechnologie<br>
# Studiengang: Geoinformationswissenschaften (M.Sc.)

# <center><img src="https://www.jade-hs.de/fileadmin/_migrated/pics/Logo_JadeHochschule_7.jpg" width="400" /></center>

# # <center>Raumzeitliche Analyse der Ausbreitung von COVID-19 in Deutschland</center>

# **<center>Masterprojekt<br>
# Wintersemester 2021/22</center>**

# Erarbeitet von: &hairsp; Ricarda Sodermanns (6020924), ricarda.sodermanns@student.jade-hs.de<br>
# &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; Nicklas Meyer (6020989), nicklas.meyer@student.jade-hs.de
# 
# &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; Oldenburg, 24. Januar 2022
# 
# Prüfer: &emsp; &emsp; &emsp; Prof. Dr. rer. nat. habil. Roland Pesch

# ## Gliederung

# * [Datenimport](#datenimport)
# * [Datensichtung](#datensichtung)
# * [Datenvorbereitung](#datenvorbereitung)
# * [Analyse](#analyse)
#     * [Verlauf](#analyse-verlauf)
#     * [Kartendarstellung der 7-Tage-Inzidenz](#analyse-map)
#     * [HotSpot-Analyse](#analyse-hsa)
#     * [Ausreißer-Analyse](#analyse-outlier)
#     * [Space Time Cubes](#analyse-stc)
#         * [Space Time Cubes berechnen](#analyse-stc-create)
#         * [Emerging Hot Spot Analyse](#analyse-stc-emerg)
#         * [Visualisierung in 3D](#analyse-stc-vis3D)
#         * [Time Series Clustering](#analyse-stc-clust)
# * [Datenexport](#export)

# Zur Durchführung des Skripts sollte in der ArcGIS-Enterprise Umgebung ein Ordner mit dem Namen 'Masterprojekt' erstellt werden, da dort zum Teil Ergebnisse abgespeichert werden.

# ### Imports

# Import der benötigten Pakete

# In[116]:


# Standard
import os
import pandas as pd
import numpy

# ArcGIS (Geofunktonalitäten)
import arcgis
from arcgis.gis import GIS
from arcgis import geoanalytics
from arcgis.features import SpatialDataFrame
from arcgis import features
import arcpy

# Matplotlib (Diagramme)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib

# Diverses
import zipfile
from copy import deepcopy
from datetime import datetime


# #### arcpy-Workspace

# Für die arcpy-Funktionen wird ein Workspace benötigt. Dieser wird im Home-Verzeichnis auf dem ArcGIS-Enterprise angelegt und dort eine Geodatabase zum Speichern der Ergebnisse erzeugt.

# In[117]:


home_dir = os.path.join(os.getcwd(), 'home')
if not arcpy.Exists(os.path.join(home_dir, 'Results.gdb')):
    arcpy.CreateFileGDB_management(home_dir, 'Results.gdb')


# In[118]:


arcpy.env.workspace = os.path.join(home_dir,'Results.gdb')
results_dir = os.path.join(home_dir,'Results.gdb')


# ## Datenimport <a class="anchor" id="datenimport"></a>

# ### Karte erstellen

# Erzeugen einer Karte zur Darstellung der Layerdaten.

# In[119]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map = gis.map("Germany")
map


# ### Layer: Kreisgrenzen

# Einladen der Kreisgrenzen als Polygondaten. Die Daten können direkt aus dem ArcGIS-Hub als Layer-Datei verwendet werden.
# Verwendet werden die Landkreisdaten des RKI mit Coronadaten, um die zu den Daten passenden Geometrien (Berlin in Stadteile aufgeteilt) sowie Einwohnerzahlen pro Landkreis zu erhalten.
# 
# Quelle: Robert Koch-Institut (RKI), dl-de/by-2-0

# In[5]:


# Item Added From Toolbar
# Title: RKI Corona Landkreise | Type: Feature Service | Owner: help1@esri
agol_gis = GIS(set_active=False)
kreise = agol_gis.content.get("917fc37a709542548cc3be077a786c17")
kreise


# Diese Daten werden in der zuvor erzeugten Karte dargestellt.

# In[6]:


map.add_layer(kreise)


# Umwandeln der Layerdaten in ein "spatially-enabled" Dataframe um mit den Daten im Notebook arbeiten zu können und aber die geografischen Informationen beizubehalten.

# In[7]:


kreise_df = pd.DataFrame.spatial.from_layer(kreise.layers[0])


# ### Corona-Daten

# Einladen der tagesaktuellen Coronadaten für den gesamten Zeitraum. Die Daten können direkt aus dem ArcGIS-Hub als CSV-Datei verwendet werden.
# 
# Quelle: Robert Koch-Institut (RKI), dl-de/by-2-0

# In[10]:


# Item Added From Toolbar
# Title: RKI_COVID19 | Type: CSV | Owner: help6@esri
agol_gis = GIS(set_active=False)
data = agol_gis.content.get("f10774f1c63e40168479a1feb6c7ca74")
data


# Umwandeln der CSV-Daten in ein Pandas-Dataframe. Dabei werden die Spalten 'Meldedatum', 'Datenstand' und 'Refdatum' als Datum abgespeichert.

# In[11]:


data_csv = data.get_data()
data_df = pd.read_csv(data_csv, parse_dates=['Meldedatum', 'Datenstand', 'Refdatum'])


# ## Datensichtung <a class="anchor" id="datensichtung"></a>

# ### Attributtabelle

# Sichten des Aufbaus der Corona-Daten durch Ausgabe der Größe, Spalten und Spaltentypen der Daten.

# In[12]:


data_df.shape


# In[13]:


data_df.columns


# In[14]:


print(data_df.dtypes)


# Sichten von Anfang und Ende des Dataframes, um Aufbau der Daten zu überprüfen.

# In[15]:


data_df.head()


# In[16]:


data_df.tail()


# ## Datenvorbereitung <a class="anchor" id="datenvorbereitung"></a>

# ### Coronadaten

# #### Aggregieren nach Tag und Landkreis

# Da in der folgenden Analyse nicht auf Alter und Geschlecht der Personen eingegangen wird, wird durch eine Aggregation der Daten nur ein Eintrag pro Tag mit der Anzahl der Fälle, Todesfälle und Genesenen erzeugt. Im Zuge dessen werden nur die für die weitere Analyse wichtigen Spalten in das neue Dataframe überführt. Außerdem wird die Landkreis-ID auch als fünf-stelliger String abgelegt.

# In[17]:


data_df_aggr = data_df[['IdBundesland','Bundesland','IdLandkreis','Landkreis','Meldedatum','AnzahlFall','AnzahlTodesfall','AnzahlGenesen']].groupby(['IdBundesland','Bundesland','IdLandkreis','Landkreis','Meldedatum']).sum()
data_df_aggr.reset_index(inplace = True, drop = False)
data_df_aggr['IdLandkreis_str'] = data_df_aggr['IdLandkreis'].astype(str).str.zfill(5)
data_df_aggr.head()


# ### Kreise

# #### AGS in Berliner Bezirke

# Berlin ist in den Coronadaten in zwölf Bezirke aufgeteilt. Die Polygondaten dazu liegen vor, allerdings ist dort der allgemeine Gemeindeschlüssel (AGS), der später zur Verbindung der Daten als Landkreis-ID dient, nicht vorhanden. Der dafür benötigte Schlüssel liegt in den Daten aber als Regionalschlüssel (RS) vor uns wird in den betreffenden Bezirken nun in die Spalte 'AGS' übertragen.

# In[18]:


for index, kreis in kreise_df.loc[kreise_df['AGS'].isnull()].iterrows():
    kreise_df.at[index, 'AGS'] = kreise_df.at[index, 'RS']
kreise_df.loc[kreise_df['BL_ID'] == '11']


# #### Geometrie und Daten trennen

# Für eine performantere Analyse werden die Geometrie und die restlichen Daten voneinander getrennt.
# 
# Endprodukte sind die Kreisgeometrien (*kreise_geom*) und die Kreisdaten mit den Einwohnerzahlen für den Landkreis (*kreise_ewz*).

# In[19]:


kreise_geom = kreise_df[['AGS', 'SHAPE', 'Shape__Area', 'Shape__Length']]
kreise_geom['AGS_int'] = kreise_geom['AGS'].astype(int)
kreise_ewz = kreise_df[['AGS', 'EWZ', 'EWZ_BL']]


# Außerdem werden die Einwohnerzahlen für die einzelnen Bundesländer in einem weiteren Dataframe (*bl_ewz*) gespeichert.
# 
# Dieses Dataframe hat pro Landkreis einen Eintrag. Damit nur noch ein Eintrag pro Bundesland bestehen bleibt, werden die Daten aggregiert (*bl_id*).

# In[20]:


bl_ewz = kreise_df[['BL_ID', 'BL', 'EWZ_BL']]
bl_ewz['BL_ID_str'] = bl_ewz['BL_ID'].astype(str).str.zfill(2)


# In[21]:


bl_id = bl_ewz[['BL_ID','BL_ID_str','BL','EWZ_BL']].groupby(['BL_ID','BL_ID_str','BL','EWZ_BL']).sum()
bl_id.reset_index(inplace = True, drop = False)
bl_id.head()


# ### Join

# Die nicht-geometrischen Kreisdaten werden mit den Coronadaten über AGS und die Landkreis-ID verbunden (*data_ewz*).

# In[22]:


data_ewz = pd.merge(kreise_ewz, data_df_aggr, left_on='AGS', right_on="IdLandkreis_str", how='right')
data_ewz.head()


# In[23]:


kreise_id = data_ewz[['IdLandkreis','IdLandkreis_str','Landkreis','EWZ','IdBundesland','Bundesland','EWZ_BL']].groupby(['IdLandkreis','IdLandkreis_str','Landkreis','EWZ','IdBundesland','Bundesland','EWZ_BL']).sum()
kreise_id.reset_index(inplace = True, drop = False)
kreise_id.head()


# #### Auf Bundesländer aggregieren

# Für eine Auswertung pro Bundesland werden die Daten auf Bundeslandebene aggregiert. Dabei werden alle Landkreisfälle pro Tag aufsummiert.

# In[24]:


data_bl = data_ewz[['IdBundesland','Bundesland','Meldedatum','AnzahlFall','AnzahlTodesfall','AnzahlGenesen', 'EWZ_BL']].groupby(['IdBundesland','Bundesland','Meldedatum', 'EWZ_BL']).sum()
data_bl.reset_index(inplace = True, drop = False)
data_bl['IdBundesland_str'] = data_bl['IdBundesland'].astype(str).str.zfill(2)
data_bl.head()


# ### Inzidenzberechnung

# Für die korrekte Berechnung der 7-Tage Inzidenz wird eine Liste aller Tage als Datum in dem zu betrachtenden Bereich erstellt. Dieser beginnt mit dem frühsten in den Daten auftrenden Fall und endet mit dem Datum des neusten Eintrag.

# In[25]:


date_list = pd.date_range(start=min(data_bl['Meldedatum']), end=max(data_bl['Meldedatum']), freq='D')
date_list


# Außerdem wird ein Dictionary mit 7 Einträgen mit 0.0 benötigt.

# In[26]:


values = [0.0,0.0,0.0,0.0,0.0,0.0,0.0] 
dict_clear = dict(zip(range(7), values))
dict_clear


# #### Bundesländer

# ##### Bundesländer: Tagesinzidenz berechnen

# Für die Inzidenzberechnung wird zunächst die Tagesinzidenz, also Fälle/Todesfälle/Genesenen pro 100.000 Einwohner berechnet. Diese Werte werden als neue Spalte dem Dataframe hinzugefügt.

# In[27]:


data_bl['FaelleEWZ'] = (data_bl['AnzahlFall'] / data_bl['EWZ_BL']) * 100000
data_bl['TodesfaelleEWZ'] = (data_bl['AnzahlTodesfall'] / data_bl['EWZ_BL']) * 100000
data_bl['GeneseneEWZ'] = (data_bl['AnzahlGenesen'] / data_bl['EWZ_BL']) * 100000
data_bl.tail()


# ##### Bundesländer: 7-Tageinzidenz berechnen

# Bei der Berechnung der 7-Tagesinzidenz muss beachtet werden, dass einige Meldedaten (besonders zu Beginn der Pandemie) nicht für alle Bundesländer vorhanden sind, weil dort keine Fälle auftraten oder es Probleme bei der Meldung der Fälle gab.
# Zunächst werden die drei dafür benötigten Spalten 'FaelleEWZ_7', 'TodesfaelleEWZ_7' und 'GeneseneEWZ_7' angelegt.
# 
# Die eigentliche Berechnung erfolgt über geschachtelte Schleifen.
# Pro Bundesland wird eine Vorlage für einen neuen Listeneintrag angelegt, falls ein neuer Eintrag durch ein fehlendes Meldedatum erstellt werden muss. Außerdem wird für die Fälle, Todesfälle und Genesene je ein Dictionary aus der zuvor angelegten Vorlage erstellt und ein Index angelegt.
# 
# Anschließend werden alle Meldedaten aus der zuvor erstellten Liste durchlaufen.
# 
# Ist ein Eintrag für den betrachteten Tag vorhanden, dann wird die aktuelle Tagesinzidenz der Fälle/Todesfälle/Genesenen in das jeweilige Dictionary an der Stelle des Tagesindex eingefügt. Außerdem werden die Spalten der 7-Tagesinzidenzen durch die Summe der im Dictionary enthaltenen Werte gefüllt.
# 
# Ist für das betrachtete Datum kein Eintrag vorhanden wird 0 in das jeweilige Dictionary an der Stelle des Tagesindex eingefügt. Außerdem wird die erstellte Vorlage mit dem  Datum und den 7-Tagesinzidenzen als Summe der im Dictionary enthaltenen Werte gefüllt. Diese Liste wird an eine Liste aller hinzuzufügenden Daten angehängt.
# 
# Sind alle Bundesländer und Tage durchlaufen wird die Liste mit den hinzuzufügenden Daten in ein Dataframe umgewandelt und an die Ursprungsdaten angehängt.

# In[28]:


data_bl['FaelleEWZ_7'] = 0.0
data_bl['TodesfaelleEWZ_7'] = 0.0
data_bl['GeneseneEWZ_7'] = 0.0

temp_data = []

for index, bl in bl_id.iterrows():

    temp_list = [bl['BL_ID'], bl['BL'],0,bl['EWZ_BL'],0,0,0,bl['BL_ID_str'],0,0,0,0,0,0]

    dict_faelle = deepcopy(dict_clear)
    dict_todesfaelle = deepcopy(dict_clear)
    dict_genesene = deepcopy(dict_clear)
    index_dict = 0

    temp_bl = data_bl.loc[data_bl['IdBundesland_str'] == bl['BL_ID_str']]

    for day in date_list:
        index_day = index_dict%7

        temp = temp_bl.loc[data_bl['Meldedatum'] == day]
        if not temp.empty:
            i = temp.index[0]
            d = data_bl.iloc[i]
            dict_faelle[index_day] = d['FaelleEWZ']
            dict_todesfaelle[index_day] = d['TodesfaelleEWZ']
            dict_genesene[index_day] = d['GeneseneEWZ']

            data_bl.at[i, 'FaelleEWZ_7'] = sum(dict_faelle.values())
            data_bl.at[i, 'TodesfaelleEWZ_7'] = sum(dict_todesfaelle.values())
            data_bl.at[i, 'GeneseneEWZ_7'] = sum(dict_genesene.values())
        else:
            dict_faelle[index_day] = 0
            dict_todesfaelle[index_day] = 0
            dict_genesene[index_day] = 0
            temp_list[2] = day
            temp_list[-3] = sum(dict_faelle.values())
            temp_list[-2] = sum(dict_todesfaelle.values())
            temp_list[-1] = sum(dict_genesene.values())
            temp_data.append(temp_list.copy())

        index_dict += 1

temp_df = pd.DataFrame(temp_data, columns=data_bl.columns)
data_bl = data_bl.append(temp_df, ignore_index=True)


# Die ergänzten Daten mit den 7-Tagesinzidenzen werden nun nach Bundesland und Meldedatum sortiert.

# In[29]:


data_bl.sort_values(['IdBundesland_str', 'Meldedatum'], inplace = True)
data_bl.reset_index(inplace = True, drop = True)


# In[30]:


data_bl.head()


# #### Landkreise

# Die Berechnung der 7-Tagesinzidenz für Landkreise funktioniert fast identisch zu der Berechnung der 7-Tagesinzidenz der Bundesländer

# ##### Landkreise: Tagesinzidenz berechnen

# Für die Inzidenzberechnung wird zunächst die Tagesinzidenz, also Fälle/Todesfälle/Genesenen pro 100.000 Einwohner berechnet. Diese Werte werden als neue Spalte dem Dataframe hinzugefügt.

# In[31]:


data_ewz['FaelleEWZ'] = (data_ewz['AnzahlFall'] / data_ewz['EWZ']) * 100000
data_ewz['TodesfaelleEWZ'] = (data_ewz['AnzahlTodesfall'] / data_ewz['EWZ']) * 100000
data_ewz['GeneseneEWZ'] = (data_ewz['AnzahlGenesen'] / data_ewz['EWZ']) * 100000
data_ewz.tail()


# #### Landkreise: 7-Tageinzidenz berechnen

# Bei der Berechnung der 7-Tagesinzidenz muss beachtet werden, dass einige Meldedaten (besonders zu Beginn der Pandemie) nicht für alle Landkreise vorhanden sind, weil dort keine Fälle auftraten oder es Probleme bei der Meldung der Fälle gab.
# Zunächst werden die drei dafür benötigten Spalten 'FaelleEWZ_7', 'TodesfaelleEWZ_7' und 'GeneseneEWZ_7' angelegt.
# 
# Die eigentliche Berechnung erfolgt über geschachtelte Schleifen.
# Pro Landkreis wird eine Vorlage für einen neuen Listeneintrag angelegt, falls ein neuer Eintrag durch ein fehlendes Meldedatum erstellt werden muss. Außerdem wird für die Fälle, Todesfälle und Genesene je ein Dictionary aus der zuvor angelegten Vorlage erstellt und ein Index angelegt.
# 
# Anschließend werden alle Meldedaten aus der zuvor erstellten Liste durchlaufen.
# 
# Ist ein Eintrag für den betrachteten Tag vorhanden, dann wird die aktuelle Tagesinzidenz der Fälle/Todesfälle/Genesenen in das jeweilige Dictionary an der Stelle des Tagesindex eingefügt. Außerdem werden die Spalten der 7-Tagesinzidenzen durch die Summe der im Dictionary enthaltenen Werte gefüllt.
# 
# Ist für das betrachtete Datum kein Eintrag vorhanden wird 0 in das jeweilige Dictionary an der Stelle des Tagesindex eingefügt. Außerdem wird die erstellte Vorlage mit dem  Datum und den 7-Tagesinzidenzen als Summe der im Dictionary enthaltenen Werte gefüllt. Diese Liste wird an eine Liste aller hinzuzufügenden Daten angehängt.
# 
# Sind alle Landkreise und Tage durchlaufen wird die Liste mit den hinzuzufügenden Daten in ein Dataframe umgewandelt und an die Ursprungsdaten angehängt.

# In[32]:


data_ewz['FaelleEWZ_7'] = 0.0
data_ewz['TodesfaelleEWZ_7'] = 0.0
data_ewz['GeneseneEWZ_7'] = 0.0

temp_data = []
bl = 0

for index, kreis in kreise_id.iterrows():

    temp_list = [kreis['IdLandkreis_str'],kreis['EWZ'],kreis['EWZ_BL'],kreis['IdBundesland'],kreis['Bundesland'],kreis['IdLandkreis'],kreis['Landkreis'],0,0,0,0,kreis['IdLandkreis_str'],0,0,0,0,0,0]

    dict_faelle = deepcopy(dict_clear)
    dict_todesfaelle = deepcopy(dict_clear)
    dict_genesene = deepcopy(dict_clear)
    index_dict = 0

    temp_kreis = data_ewz.loc[data_ewz['AGS'] == kreis['IdLandkreis_str']]

    for day in date_list:
        index_day = index_dict%7

        temp = temp_kreis.loc[data_ewz['Meldedatum'] == day]
        if not temp.empty:
            i = temp.index[0]
            d = data_ewz.iloc[i]
            dict_faelle[index_day] = d['FaelleEWZ']
            dict_todesfaelle[index_day] = d['TodesfaelleEWZ']
            dict_genesene[index_day] = d['GeneseneEWZ']

            data_ewz.at[i, 'FaelleEWZ_7'] = sum(dict_faelle.values())
            data_ewz.at[i, 'TodesfaelleEWZ_7'] = sum(dict_todesfaelle.values())
            data_ewz.at[i, 'GeneseneEWZ_7'] = sum(dict_genesene.values())
        else:
            dict_faelle[index_day] = 0
            dict_todesfaelle[index_day] = 0
            dict_genesene[index_day] = 0
            temp_list[7] = day
            temp_list[-3] = sum(dict_faelle.values())
            temp_list[-2] = sum(dict_todesfaelle.values())
            temp_list[-1] = sum(dict_genesene.values())
            temp_data.append(temp_list.copy())

        index_dict += 1
temp_df = pd.DataFrame(temp_data, columns=data_ewz.columns)
data_ewz = data_ewz.append(temp_df, ignore_index=True)


# Die ergänzten Daten mit den 7-Tagesinzidenzen werden nun nach Landkreis und Meldedatum sortiert.

# In[33]:


data_ewz.sort_values(['IdLandkreis_str', 'Meldedatum'], inplace = True)
data_ewz.reset_index(inplace = True, drop = True)


# In[34]:


data_ewz.tail()


# Zwischenspeichern der Landkreisdaten mit 7-Tages-Inzidenz, um eine Neuberechung umgehen zu können.

# In[35]:


data_ewz.to_csv('home/data.csv', index=False)


# In[36]:


data_ewz = pd.read_csv('home/data.csv', parse_dates=['Meldedatum'], dtype={'AGS': str, 'IdLandkreis_str': str})


# In[37]:


data_ewz.tail()


# ## Analyse <a class="anchor" id="analyse"></a>

# ### Übersichtskarte

# #### Daten auf Geometrie joinen

# Joinen der Daten eines Tages mit den Geometrien, um eine Übersichtskarte erstellen zu können.

# In[38]:


data_kreise_day = pd.merge(data_ewz.loc[data_ewz['Meldedatum']=='2021-12-28'], kreise_geom, left_on="IdLandkreis_str", right_on="AGS", how='right')
data_kreise_day.head()


# Erstellen eines anzeigbaren Layers für die Karte.

# In[39]:


gis = GIS("home")
items = gis.content.search(query ='Corona-Übersicht')
for item in items:
    item.delete()
data_kreise_day_fl = data_kreise_day.spatial.to_featurelayer('Corona-Übersicht', tags=['Corona', 'COVID-19'], folder='Masterprojekt')


# #### In Karte anzeigen

# Anzeigen der Karte.
# 
# Dabei kann beim ersten Ausführen ein Fehler auftreten. Um diesen zu beheben muss die Zelle einmal mit gis=GIS("home") ausgeführt und die darauffolgende Zeile auskommentiert werden werden. Anschließend kann wieder die erste Zeile auskommentiert werden und die Zelle mit gis=GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis") ausgeführt werden.

# In[41]:


#gis = GIS("home")
gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map = gis.map("Germany")
classed_color_renderer =  {"renderer":"ClassedColorRenderer", "field_name":"anzahl_fall" }
map.add_layer(data_kreise_day_fl, classed_color_renderer)
map


# ### Verlauf <a class="anchor" id="analyse-verlauf"></a>

# Um ein Verlaufdiagramm der Fälle/Todesfälle/Genesenen pro Bundesland zu erstellen muss zunächst eine Farbpalette erstellt werden und die Rahmenbedingungen gesetzt werden. Das Enddatum für das Diagramm wird auf den 02.01.2022 gesetzt. Anschließend werden alle Bundesländer nacheinander geplottet. Als letzter Schritt müssen alle Einstellungen des Diagramms gesetzt werden. Das Diagramm wird als PDF exportiert.

# #### Diagramm Fälle

# In[42]:


# Vorbereitung
col = {'01':'blue', '02':'dodgerblue', '03':'cyan', '04':'lime', '05':'forestgreen', '06':'yellow', '07':'gold', '08':'darkorange', '09':'saddlebrown', '10':'red', '11':'darkred', '12':'deeppink', '13':'darkmagenta', '14':'mediumorchid', '15':'silver', '16':'midnightblue'}
fig, ax = plt.subplots(figsize=(35,15))
y_max = max(data_bl['FaelleEWZ_7'])
x_min = min(data_bl['Meldedatum'])
x_max = datetime.strptime("02/01/22", '%d/%m/%y')

# Daten
for bl in bl_id['BL_ID_str']:
    plt.plot( data_bl.loc[data_bl['IdBundesland_str'] == bl]['Meldedatum'], data_bl.loc[data_bl['IdBundesland_str'] == bl]['FaelleEWZ_7'], color=col[bl], linewidth=3, label=bl_id.loc[bl_id['BL_ID_str'] == bl]['BL'].item())

# Einstellungen
plt.rc("font", size = 22)
plt.rc("axes", labelsize = 20)
plt.rc("legend", fontsize = 22)

plt.legend(loc='upper left')
plt.xlabel("Meldedatum")
plt.ylabel("COVID-19-Fälle der letzten 7 Tage/100.000 Einwohner")
plt.title("7-Tageinzidenz der COVID-19-Fälle pro Bundesland")
plt.xlim([x_min, x_max])
plt.ylim([0, y_max+10])
plt.xticks(rotation=45, ha='right')

ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

handles, labels = plt.gca().get_legend_handles_labels()
order = [0,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

plt.savefig('VerlaufFaelle.pdf')  


# #### Diagramm Todesfälle

# In[43]:


# Vorbereitung
col = {'01':'blue', '02':'dodgerblue', '03':'cyan', '04':'lime', '05':'forestgreen', '06':'yellow', '07':'gold', '08':'darkorange', '09':'saddlebrown', '10':'red', '11':'darkred', '12':'deeppink', '13':'darkmagenta', '14':'mediumorchid', '15':'silver', '16':'midnightblue'}
fig, ax = plt.subplots(figsize=(35,15))
y_max = max(data_bl['TodesfaelleEWZ_7'])
x_min = min(data_bl['Meldedatum'])
x_max = datetime.strptime("02/01/22", '%d/%m/%y')

# Daten
for bl in bl_id['BL_ID_str']:
    plt.plot( data_bl.loc[data_bl['IdBundesland_str'] == bl]['Meldedatum'], data_bl.loc[data_bl['IdBundesland_str'] == bl]['TodesfaelleEWZ_7'], color=col[bl], linewidth=3, label=bl_id.loc[bl_id['BL_ID_str'] == bl]['BL'].item())

# Einstellungen
plt.legend(loc='upper left')
plt.xlabel("Meldedatum der Erkrankung")
plt.ylabel("COVID-19-Todesfälle der letzten 7 Tage/100.000 Einwohner")
plt.title("7-Tageinzidenz der COVID-19-Todesfälle pro Bundesland")
plt.xlim([x_min, x_max])
plt.ylim([0, y_max+0.5])
plt.xticks(rotation=45, ha='right')

ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

handles, labels = plt.gca().get_legend_handles_labels()
order = [0,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

plt.savefig('VerlaufTodesfaelle.pdf')  


# #### Diagramm Genesene

# In[44]:


# Vorbereitung
col = {'01':'blue', '02':'dodgerblue', '03':'cyan', '04':'lime', '05':'forestgreen', '06':'yellow', '07':'gold', '08':'darkorange', '09':'saddlebrown', '10':'red', '11':'darkred', '12':'deeppink', '13':'darkmagenta', '14':'mediumorchid', '15':'silver', '16':'midnightblue'}
fig, ax = plt.subplots(figsize=(35,15))
y_max = max(data_bl['GeneseneEWZ_7'])
x_min = min(data_bl['Meldedatum'])
x_max = datetime.strptime("02/01/22", '%d/%m/%y')

# Daten
for bl in bl_id['BL_ID_str']:
    plt.plot( data_bl.loc[data_bl['IdBundesland_str'] == bl]['Meldedatum'], data_bl.loc[data_bl['IdBundesland_str'] == bl]['GeneseneEWZ_7'], color=col[bl], linewidth=3, label=bl_id.loc[bl_id['BL_ID_str'] == bl]['BL'].item())

# Einstellungen
plt.legend(loc='upper left')
plt.xlabel("Meldedatum der Erkrankung")
plt.ylabel("COVID-19-Genesene der letzten 7 Tage/100.000 Einwohner")
plt.title("7-Tageinzidenz der COVID-19-Genesene pro Bundesland")
plt.xlim([x_min, x_max])
plt.ylim([0, y_max+5])
plt.xticks(rotation=45, ha='right')

ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

handles, labels = plt.gca().get_legend_handles_labels()
order = [0,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 

plt.savefig('VerlaufGenesene.pdf')  


# ### Kartendarstellung der 7-Tage-Inzidenz <a class="anchor" id="analyse-map"></a>

# Die 7-Tageinzidenz soll für alle vier Wellenhochpunkte dargestellt werden.
# 
# - 1. Welle: 16.03.2020
# - 2. Welle: 16.12.2020
# - 3. Welle: 21.04.2021
# - 4. Welle: 24.11.2021

# #### Maximum 1. Welle (16.03.2020)

# Zunächst müssen die Daten für den Hochpunkt der 1. Welle am 16.03.2020 herausgefiltert und mit den Geometriedaten verknüpft werden.

# In[45]:


data_Max1W = data_ewz.loc[data_ewz['Meldedatum']=='2020-03-16']
data_Max1W.head()


# In[46]:


data_Max1W_Geom = pd.merge(data_Max1W, kreise_geom, left_on="IdLandkreis_str", right_on="AGS", how='right')
data_Max1W_Geom.head()


# Um die Daten darstellen zu können muss ein Layer in dem Projektordner erzeugt werden. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.

# In[47]:


gis = GIS("home")
items = gis.content.search(query ='Corona-7Tageinzidenz1W')
for item in items:
    item.delete()
data_Max1W_fl = data_Max1W_Geom.spatial.to_featurelayer('Corona-7Tageinzidenz1W', tags=['Corona', 'COVID-19'], folder='Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[48]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_Max1W = gis.map("Germany")
classed_color_renderer =  {"renderer":"ClassedColorRenderer", "field_name":"faelle_ewz_7" }
map_Max1W.add_layer(data_Max1W_fl, classed_color_renderer)
map_Max1W


# #### Maximum 2. Welle (16.12.2020)

# Zunächst müssen die Daten für den Hochpunkt der 2. Welle am 16.12.2020 herausgefiltert und mit den Geometriedaten verknüpft werden.

# In[49]:


data_Max2W = data_ewz.loc[data_ewz['Meldedatum']=='2020-12-16']
data_Max2W.head()


# In[50]:


data_Max2W_Geom = pd.merge(data_Max2W, kreise_geom, left_on="IdLandkreis_str", right_on="AGS", how='right')
data_Max2W_Geom.head()


# Um die Daten darstellen zu können muss ein Layer in dem Projektordner erzeugt werden. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.

# In[51]:


gis = GIS("home")
items = gis.content.search(query ='Corona-7Tageinzidenz2W')
for item in items:
    item.delete()
data_Max2W_fl = data_Max2W_Geom.spatial.to_featurelayer('Corona-7Tageinzidenz2W', tags=['Corona', 'COVID-19'], folder='Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[52]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_Max2W = gis.map("Germany")
classed_color_renderer =  {"renderer":"ClassedColorRenderer", "field_name":"faelle_ewz_7" }
map_Max2W.add_layer(data_Max2W_fl, classed_color_renderer)
map_Max2W


# #### Maximum 3. Welle (21.04.2021)

# Zunächst müssen die Daten für den Hochpunkt der 2. Welle am 21.04.2021 herausgefiltert und mit den Geometriedaten verknüpft werden.

# In[53]:


data_Max3W = data_ewz.loc[data_ewz['Meldedatum']=='2021-04-21']
data_Max3W.head()


# In[54]:


data_Max3W_Geom = pd.merge(data_Max3W, kreise_geom, left_on="IdLandkreis_str", right_on="AGS", how='right')
data_Max3W_Geom.head()


# Um die Daten darstellen zu können muss ein Layer in dem Projektordner erzeugt werden. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.

# In[55]:


gis = GIS("home")
items = gis.content.search(query ='Corona-7Tageinzidenz3W')
for item in items:
    item.delete()
data_Max3W_fl = data_Max3W_Geom.spatial.to_featurelayer('Corona-7Tageinzidenz3W', tags=['Corona', 'COVID-19'], folder='Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[56]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_Max3W = gis.map("Germany")
classed_color_renderer =  {"renderer":"ClassedColorRenderer", "field_name":"faelle_ewz_7" }
map_Max3W.add_layer(data_Max3W_fl, classed_color_renderer)
map_Max3W


# #### Maximum 4. Welle (24.11.2021)

# Zunächst müssen die Daten für den Hochpunkt der 4. Welle am 24.11.2021 herausgefiltert und mit den Geometriedaten verknüpft werden.

# In[57]:


data_Max4W = data_ewz.loc[data_ewz['Meldedatum']=='2021-11-24']
data_Max4W.head()


# In[58]:


data_Max4W_Geom = pd.merge(data_Max4W, kreise_geom, left_on="IdLandkreis_str", right_on="AGS", how='right')
data_Max4W_Geom.head()


# Um die Daten darstellen zu können muss ein Layer in dem Projektordner erzeugt werden. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.

# In[59]:


gis = GIS("home")
items = gis.content.search(query ='Corona-7Tageinzidenz4W')
for item in items:
    item.delete()
data_Max4W_fl = data_Max4W_Geom.spatial.to_featurelayer('Corona-7Tageinzidenz4W', tags=['Corona', 'COVID-19'], folder='Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[60]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_Max4W = gis.map("Germany")
classed_color_renderer =  {"renderer":"ClassedColorRenderer", "field_name":"faelle_ewz_7" }
map_Max4W.add_layer(data_Max4W_fl, classed_color_renderer)
map_Max4W


# ### HotSpot-Analyse <a class="anchor" id="analyse-hsa"></a>

# Es soll eine Hot Spot-Analyse für alle vier Wellenhochpunkte durchgeführt werden.
# 
# - 1. Welle: 16.03.2020
# - 2. Welle: 16.12.2020
# - 3. Welle: 21.04.2021
# - 4. Welle: 24.11.2021

# Eine Erklärung des Werkzeugs von ArcGIS ist unter diesem Link zu finden:
# https://developers.arcgis.com/python/api-reference/arcgis.features.analyze_patterns.html#find-hot-spots

# #### Maximum 1. Welle (16.03.2020)

# Auch hier wird bei der Analyse ein Layer erzeugt. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.
# 
# Nach der Durchführung der Hot Spot-Analyse über **arcgis.features.analyze_patterns.find_hot_spots()** wird der Ergebnislayer noch in den Projektordner verschoben.

# In[61]:


gis = GIS("home")
items = gis.content.search(query ='HotSpot_1W')
for item in items:
    item.delete()
hotspot_1W = arcgis.features.analyze_patterns.find_hot_spots(data_Max1W_fl, analysis_field="faelle_ewz_7", output_name="HotSpot_1W", distance_band=None, distance_band_unit=None)
hotspot_1W.move('Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[62]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_HS_1W = gis.map("Germany")
map_HS_1W.add_layer(hotspot_1W)
map_HS_1W


# #### Maximum 2. Welle (16.12.2020)

# Auch hier wird bei der Analyse ein Layer erzeugt. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.
# 
# Nach der Durchführung der Hot Spot-Analyse über **arcgis.features.analyze_patterns.find_hot_spots()** wird der Ergebnislayer noch in den Projektordner verschoben.

# In[63]:


gis = GIS("home")
items = gis.content.search(query ='HotSpot_2W')
for item in items:
    item.delete()
hotspot_2W = arcgis.features.analyze_patterns.find_hot_spots(data_Max2W_fl, analysis_field="faelle_ewz_7", output_name="HotSpot_2W", distance_band=None, distance_band_unit=None)
hotspot_2W.move('Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[64]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_HS_2W = gis.map("Germany")
map_HS_2W.add_layer(hotspot_2W)
map_HS_2W


# #### Maximum 3. Welle (21.04.2021)

# Auch hier wird bei der Analyse ein Layer erzeugt. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.
# 
# Nach der Durchführung der Hot Spot-Analyse über **arcgis.features.analyze_patterns.find_hot_spots()** wird der Ergebnislayer noch in den Projektordner verschoben.

# In[65]:


gis = GIS("home")
items = gis.content.search(query ='HotSpot_3W')
for item in items:
    item.delete()
hotspot_3W = arcgis.features.analyze_patterns.find_hot_spots(data_Max3W_fl, analysis_field="faelle_ewz_7", output_name="HotSpot_3W", distance_band=None, distance_band_unit=None)
hotspot_3W.move('Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[66]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_HS_3W = gis.map("Germany")
map_HS_3W.add_layer(hotspot_3W)
map_HS_3W


# #### Maximum 4. Welle (24.11.2021)

# Auch hier wird bei der Analyse ein Layer erzeugt. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.
# 
# Nach der Durchführung der Hot Spot-Analyse über **arcgis.features.analyze_patterns.find_hot_spots()** wird der Ergebnislayer noch in den Projektordner verschoben.

# In[67]:


gis = GIS("home")
items = gis.content.search(query ='HotSpot_4W')
for item in items:
    item.delete()
hotspot_4W = arcgis.features.analyze_patterns.find_hot_spots(data_Max4W_fl, analysis_field="faelle_ewz_7", output_name="HotSpot_4W", distance_band=None, distance_band_unit=None)
hotspot_4W.move('Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[68]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_HS_4W = gis.map("Germany")
map_HS_4W.add_layer(hotspot_4W)
map_HS_4W


# ### Ausreißer-Analyse <a class="anchor" id="analyse-outlier"></a>

# Es soll eine Ausreißer-Analyse für alle vier Wellenhochpunkte durchgeführt werden.
# 
# - 1. Welle: 16.03.2020
# - 2. Welle: 16.12.2020
# - 3. Welle: 21.04.2021
# - 4. Welle: 24.11.2021

# #### 1. Welle (16.03.2020)

# Auch hier wird bei der Analyse ein Layer erzeugt. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.
# 
# Nach der Durchführung der Ausreißer-Analyse über **arcgis.features.analyze_patterns.find_outliers()** wird der Ergebnislayer noch in den Projektordner verschoben.

# In[69]:


gis = GIS("home")
items = gis.content.search(query ='Outliers_1W')
for item in items:
    item.delete()
outliers_1W_Res = arcgis.features.analyze_patterns.find_outliers(data_Max1W_fl, analysis_field="faelle_ewz_7", output_name="Outliers_1W")
outliers_1W = outliers_1W_Res['outliers_result_layer']
outliers_1W.move('Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[70]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_Out_1W = gis.map("Germany")
map_Out_1W.add_layer(outliers_1W)
map_Out_1W


# #### 2. Welle (16.12.2020)

# Auch hier wird bei der Analyse ein Layer erzeugt. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.
# 
# Nach der Durchführung der Ausreißer-Analyse über **arcgis.features.analyze_patterns.find_outliers()** wird der Ergebnislayer noch in den Projektordner verschoben.

# In[71]:


gis = GIS("home")
items = gis.content.search(query ='Outliers_2W')
for item in items:
    item.delete()
outliers_2W_Res = arcgis.features.analyze_patterns.find_outliers(data_Max2W_fl, analysis_field="faelle_ewz_7", output_name="Outliers_2W")
outliers_2W = outliers_2W_Res['outliers_result_layer']
outliers_2W.move('Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[72]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_Out_2W = gis.map("Germany")
map_Out_2W.add_layer(outliers_2W)
map_Out_2W


# #### 3. Welle (21.04.2021)

# Auch hier wird bei der Analyse ein Layer erzeugt. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.
# 
# Nach der Durchführung der Ausreißer-Analyse über **arcgis.features.analyze_patterns.find_outliers()** wird der Ergebnislayer noch in den Projektordner verschoben.

# In[73]:


gis = GIS("home")
items = gis.content.search(query ='Outliers_3W')
for item in items:
    item.delete()
outliers_3W_Res = arcgis.features.analyze_patterns.find_outliers(data_Max3W_fl, analysis_field="faelle_ewz_7", output_name="Outliers_3W")
outliers_3W = outliers_3W_Res['outliers_result_layer']
outliers_3W.move('Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[74]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_Out_3W = gis.map("Germany")
map_Out_3W.add_layer(outliers_3W)
map_Out_3W


# #### 4. Welle (24.11.2021)

# Auch hier wird bei der Analyse ein Layer erzeugt. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.
# 
# Nach der Durchführung der Ausreißer-Analyse über **arcgis.features.analyze_patterns.find_outliers()** wird der Ergebnislayer noch in den Projektordner verschoben.

# In[75]:


gis = GIS("home")
items = gis.content.search(query ='Outliers_4W')
for item in items:
    item.delete()
outliers_4W_Res = arcgis.features.analyze_patterns.find_outliers(data_Max4W_fl, analysis_field="faelle_ewz_7", output_name="Outliers_4W")
outliers_4W = outliers_4W_Res['outliers_result_layer']
outliers_4W.move('Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[76]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_Out_4W = gis.map("Germany")
map_Out_4W.add_layer(outliers_4W)
map_Out_4W


# ### Space-Time Cubes <a class="anchor" id="analyse-stc"></a>

# Es sollen Space-Time Cubes für alle vier Wellen durchgeführt werden.
# 
# - 1. Welle: 02.03.2020 - 19.04.2020
# - 2. Welle: 05.10.2020 - 31.01.2021
# - 3. Welle: 01.03.2021 - 16.05.2021
# - 4. Welle: 04.10.2021 - 02.01.2022
# 
# Dabei werden arcpy-Funktionen verwendet. Dafür müssen die Kreisgeometrien zunächst in der Geodatabase abgespeichert werden.

# In[77]:


path_geom = os.path.join(results_dir, 'kreise_geom')
kreise_geom.spatial.to_featureclass(path_geom)


# #### Space Time Cubes berechnen <a class="anchor" id="analyse-stc-create"></a>
# 
# Die Funktionsweise der Space-Time Cubes kann unter folgenden Link nachvollzogen werden: 
# https://pro.arcgis.com/de/pro-app/latest/tool-reference/space-time-pattern-mining/createcubefromdefinedlocations.htm

# ##### kleine Datenmenge (10.12.2020 - 24.12.2020)

# Testweise wurden der Space-Time Cube für eine kleiner Datenmenge von zwei Wochen (10.12.2020 - 24.12.2020) durchgeführt. Dieser Zeitraum muss aus den Daten extrahiert und ebenfalls in der Geodatabase als Tabelle abgespeichert werden.

# In[78]:


data_stc_test = data_ewz.loc[(data_ewz['Meldedatum']>='2020-12-10') & (data_ewz['Meldedatum']<='2020-12-24')]
data_stc_test


# In[79]:


path_test = os.path.join(results_dir, 'data_test')
data_stc_test.spatial.to_table(path_test)


# Falls der zu erstellende Space-Time Cube aus vorherigen Durchläufen bereits vorliegt wird dieser gelöscht.
# 
# Da die Daten an festgelegten Position vorliegen, wird der Space-Time Cubes über die arcpy Funktion **arcpy.stpm.CreateSpaceTimeCubeDefinedLocations()** berechnet und in der Geodatabase abgelegt.

# In[80]:


arcpy.management.Delete(os.path.join(home_dir, 'stc_Test.nc'))
stc_test = arcpy.stpm.CreateSpaceTimeCubeDefinedLocations(path_geom, os.path.join(home_dir, 'stc_Test.nc'), 'AGS_int','NO_TEMPORAL_AGGREGATION', 'meldedatum', '1 Days', '', '', 'FaelleEWZ_7 ZEROS', '', path_test, 'IdLandkreis')


# ##### 1. Welle (02.03.2020 - 19.04.2020)

# Die Daten der ersten Welle müssen aus den gesamten Daten extrahiert und ebenfalls in der Geodatabase als Tabelle abgespeichert werden.

# In[81]:


data_stc_1W = data_ewz.loc[(data_ewz['Meldedatum']>='2020-03-02') & (data_ewz['Meldedatum']<='2020-04-19')]
data_stc_1W


# In[82]:


path_1W = os.path.join(results_dir, 'data_1W')
data_stc_1W.spatial.to_table(path_1W)


# Falls der zu erstellende Space-Time Cube aus vorherigen Durchläufen bereits vorliegt wird dieser gelöscht.
# 
# Da die Daten an festgelegten Position vorliegen, wird der Space-Time Cubes über die arcpy Funktion **arcpy.stpm.CreateSpaceTimeCubeDefinedLocations()** berechnet und in der Geodatabase abgelegt.

# In[83]:


arcpy.management.Delete(os.path.join(home_dir, 'stc_1W.nc'))
stc_1W = arcpy.stpm.CreateSpaceTimeCubeDefinedLocations(path_geom, os.path.join(home_dir, 'stc_1W.nc'), 'AGS_int', 'APPLY_TEMPORAL_AGGREGATION', 'meldedatum', '3 Days', 'END_TIME', '', '','FaelleEWZ_7 MEAN ZEROS', path_1W, 'IdLandkreis')


# ##### 2. Welle (05.10.2020 - 31.01.2021)

# Die Daten der zweiten Welle müssen aus den gesamten Daten extrahiert und ebenfalls in der Geodatabase als Tabelle abgespeichert werden.

# In[84]:


data_stc_2W = data_ewz.loc[(data_ewz['Meldedatum']>='2020-10-05') & (data_ewz['Meldedatum']<='2021-01-31')]
data_stc_2W


# In[85]:


path_2W = os.path.join(results_dir, 'data_2W')
data_stc_2W.spatial.to_table(path_2W)


# Falls der zu erstellende Space-Time Cube aus vorherigen Durchläufen bereits vorliegt wird dieser gelöscht.
# 
# Da die Daten an festgelegten Position vorliegen, wird der Space-Time Cubes über die arcpy Funktion **arcpy.stpm.CreateSpaceTimeCubeDefinedLocations()** berechnet und in der Geodatabase abgelegt.

# In[86]:


arcpy.management.Delete(os.path.join(home_dir, 'stc_2W.nc'))
stc_2W = arcpy.stpm.CreateSpaceTimeCubeDefinedLocations(path_geom, os.path.join(home_dir, 'stc_2W.nc'), 'AGS_int', 'APPLY_TEMPORAL_AGGREGATION', 'meldedatum', '1 Weeks', 'END_TIME', '', '','FaelleEWZ_7 MEAN ZEROS', path_2W, 'IdLandkreis')


# ##### 3. Welle (01.03.2021 - 16.05.2021)

# Die Daten der dritten Welle müssen aus den gesamten Daten extrahiert und ebenfalls in der Geodatabase als Tabelle abgespeichert werden.

# In[87]:


data_stc_3W = data_ewz.loc[(data_ewz['Meldedatum']>='2021-03-01') & (data_ewz['Meldedatum']<='2021-05-16')]
data_stc_3W


# In[88]:


path_3W = os.path.join(results_dir, 'data_3W')
data_stc_3W.spatial.to_table(path_3W)


# Falls der zu erstellende Space-Time Cube aus vorherigen Durchläufen bereits vorliegt wird dieser gelöscht.
# 
# Da die Daten an festgelegten Position vorliegen, wird der Space-Time Cubes über die arcpy Funktion **arcpy.stpm.CreateSpaceTimeCubeDefinedLocations()** berechnet und in der Geodatabase abgelegt.

# In[89]:


arcpy.management.Delete(os.path.join(home_dir, 'stc_3W.nc'))
stc_3W = arcpy.stpm.CreateSpaceTimeCubeDefinedLocations(path_geom, os.path.join(home_dir, 'stc_3W.nc'), 'AGS_int', 'APPLY_TEMPORAL_AGGREGATION', 'meldedatum', '1 Weeks', 'END_TIME', '', '','FaelleEWZ_7 MEAN ZEROS', path_3W, 'IdLandkreis')


# ##### 4. Welle (04.10.2021 - 02.01.2022)

# Die Daten der vierten Welle müssen aus den gesamten Daten extrahiert und ebenfalls in der Geodatabase als Tabelle abgespeichert werden.

# In[90]:


data_stc_4W = data_ewz.loc[(data_ewz['Meldedatum']>='2021-10-04') & (data_ewz['Meldedatum']<='2022-01-02')]
data_stc_4W


# In[91]:


path_4W = os.path.join(results_dir, 'data_4W')
data_stc_4W.spatial.to_table(path_4W)


# Falls der zu erstellende Space-Time Cube aus vorherigen Durchläufen bereits vorliegt wird dieser gelöscht.
# 
# Da die Daten an festgelegten Position vorliegen, wird der Space-Time Cubes über die arcpy Funktion **arcpy.stpm.CreateSpaceTimeCubeDefinedLocations()** berechnet und in der Geodatabase abgelegt.

# In[ ]:


arcpy.management.Delete(os.path.join(home_dir, 'stc_4W.nc'))
stc_4W = arcpy.stpm.CreateSpaceTimeCubeDefinedLocations(path_geom, os.path.join(home_dir, 'stc_4W.nc'), 'AGS_int', 'APPLY_TEMPORAL_AGGREGATION', 'meldedatum', '1 Weeks', 'END_TIME', '', '','FaelleEWZ_7 MEAN ZEROS', path_4W, 'IdLandkreis')


# #### Emerging Hot Spot Analyse <a class="anchor" id="analyse-stc-emerg"></a>
# 
# Mit Hilfe der zuvor erstellten Space-Time Cubes soll nun eine Emerging Hot Spot-Analyse durchgeführt werden.
# 
# Die Funktionsweise des Werkzeugs kann hier nachvollzogen werden:
# https://pro.arcgis.com/de/pro-app/latest/tool-reference/space-time-pattern-mining/emerginghotspots.htm

# ##### kleine Datenmenge (10.12.2020 - 24.12.2020)

# Falls das Ergebnis der zu erstellenden Emerging Hot Spot-Analyse aus vorherigen Durchläufen bereits vorliegt wird dieses gelöscht.
# 
# Die Emerging Hot Spot-Analyse wird über die arcpy-Funktion **arcpy.stpm.EmergingHotSpotAnalysis()** mit Hilfe der Space-Time Cubes berechnet und in der Geodatabase abgelegt.

# In[120]:


arcpy.management.Delete(os.path.join(results_dir, 'emerg_Test'))
emerg = arcpy.stpm.EmergingHotSpotAnalysis(os.path.join(home_dir, 'stc_Test.nc'), "FAELLEEWZ_7_NONE_ZEROS", os.path.join(results_dir, 'emerg_Test'))


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um im ArcGIS-Enterprise ein Layer erzeugen zu können.

# In[121]:


emerg_Test_SDF = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'Emerg_Test'))


# Um die Daten darstellen zu können muss ein Layer in dem Projektordner erzeugt werden. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.

# In[122]:


gis = GIS("home")
items = gis.content.search(query ='Emerg_Test')
for item in items:
    item.delete()
emerg_test = emerg_Test_SDF.spatial.to_featurelayer('Emerg_Test', tags=['Corona', 'Covid-19'], folder='Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[125]:


#gis = GIS("home")
gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_Emerg_Test = gis.map("Germany")
classed_color_renderer =  {"renderer":"ClassedColorRenderer", "field_name":"category" }
#vis3D_Test_SEDF.spatial.plot(vis3D_Test_map, col="HS_BIN", cmap='bwr', renderer_type='c', method='esriClassifyNaturalBreaks', min_value=-3, class_count=7, alpha=0.8, line_width=0.2)

map_Emerg_Test.add_layer(emerg_test, classed_color_renderer)
map_Emerg_Test


# ##### 1. Welle (02.03.2020 - 19.04.2020)

# Falls das Ergebnis der zu erstellenden Emerging Hot Spot-Analyse aus vorherigen Durchläufen bereits vorliegt wird dieses gelöscht.
# 
# Die Emerging Hot Spot-Analyse wird über die arcpy Funktion **arcpy.stpm.EmergingHotSpotAnalysis()** mit Hilfe der Space-Time Cubes berechnet und in der Geodatabase abgelegt.

# In[126]:


arcpy.management.Delete(os.path.join(results_dir, 'emerg_1W'))
emerg = arcpy.stpm.EmergingHotSpotAnalysis(os.path.join(home_dir, 'stc_1W.nc'), "FAELLEEWZ_7_MEAN_ZEROS", os.path.join(results_dir, 'emerg_1W'))


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um im ArcGIS-Enterprise ein Layer erzeugen zu können.

# In[127]:


emerg_1W_SDF = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'emerg_1W'))


# Um die Daten darstellen zu können muss ein Layer in dem Projektordner erzeugt werden. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.

# In[128]:


gis = GIS("home")
items = gis.content.search(query ='emerg_1W')
for item in items:
    item.delete()
emerg_1W = emerg_1W_SDF.spatial.to_featurelayer('emerg_1W', tags=['Corona', 'Covid-19'], folder='Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[129]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_emerg_1W = gis.map("Germany")
classed_color_renderer =  {"renderer":"ClassedColorRenderer", "field_name":"category" }
map_emerg_1W.add_layer(emerg_1W, classed_color_renderer)
map_emerg_1W


# ##### 2. Welle (05.10.2020 - 31.01.2021)

# Falls das Ergebnis der zu erstellenden Emerging Hot Spot-Analyse aus vorherigen Durchläufen bereits vorliegt wird dieses gelöscht.
# 
# Die Emerging Hot Spot-Analyse wird über die arcpy Funktion **arcpy.stpm.EmergingHotSpotAnalysis()** mit Hilfe der Space-Time Cubes berechnet und in der Geodatabase abgelegt.

# In[130]:


arcpy.management.Delete(os.path.join(results_dir, 'emerg_2W'))
emerg = arcpy.stpm.EmergingHotSpotAnalysis(os.path.join(home_dir, 'stc_2W.nc'), "FAELLEEWZ_7_MEAN_ZEROS", os.path.join(results_dir, 'emerg_2W'))


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um im ArcGIS-Enterprise ein Layer erzeugen zu können.

# In[131]:


emerg_2W_SDF = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'emerg_2W'))


# Um die Daten darstellen zu können muss ein Layer in dem Projektordner erzeugt werden. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.

# In[132]:


gis = GIS("home")
items = gis.content.search(query ='emerg_2W')
for item in items:
    item.delete()
emerg_2W = emerg_2W_SDF.spatial.to_featurelayer('emerg_2W', tags=['Corona', 'Covid-19'], folder='Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[133]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_emerg_2W = gis.map("Germany")
classed_color_renderer =  {"renderer":"ClassedColorRenderer", "field_name":"category" }
map_emerg_2W.add_layer(emerg_2W, classed_color_renderer)
map_emerg_2W


# ##### 3. Welle (01.03.2021 - 16.05.2021)

# Falls das Ergebnis der zu erstellenden Emerging Hot Spot-Analyse aus vorherigen Durchläufen bereits vorliegt wird dieses gelöscht.
# 
# Die Emerging Hot Spot-Analyse wird über die arcpy Funktion **arcpy.stpm.EmergingHotSpotAnalysis()** mit Hilfe der Space-Time Cubes berechnet und in der Geodatabase abgelegt.

# In[134]:


arcpy.management.Delete(os.path.join(results_dir, 'emerg_3W'))
emerg = arcpy.stpm.EmergingHotSpotAnalysis(os.path.join(home_dir, 'stc_3W.nc'), "FAELLEEWZ_7_MEAN_ZEROS", os.path.join(results_dir, 'emerg_3W'))


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um im ArcGIS-Enterprise ein Layer erzeugen zu können.

# In[135]:


emerg_3W_SDF = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'emerg_3W'))


# Um die Daten darstellen zu können muss ein Layer in dem Projektordner erzeugt werden. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.

# In[136]:


gis = GIS("home")
items = gis.content.search(query ='emerg_3W')
for item in items:
    item.delete()
emerg_3W = emerg_3W_SDF.spatial.to_featurelayer('emerg_3W', tags=['Corona', 'Covid-19'], folder='Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[137]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_emerg_3W = gis.map("Germany")
classed_color_renderer =  {"renderer":"ClassedColorRenderer", "field_name":"category" }
map_emerg_3W.add_layer(emerg_3W, classed_color_renderer)
map_emerg_3W


# ##### 4. Welle (04.10.2021 - 02.01.2022)

# Falls das Ergebnis der zu erstellenden Emerging Hot Spot-Analyse aus vorherigen Durchläufen bereits vorliegt wird dieses gelöscht.
# 
# Die Emerging Hot Spot-Analyse wird über die arcpy Funktion **arcpy.stpm.EmergingHotSpotAnalysis()** mit Hilfe der Space-Time Cubes berechnet und in der Geodatabase abgelegt.

# In[138]:


arcpy.management.Delete(os.path.join(results_dir, 'emerg_4W'))
emerg = arcpy.stpm.EmergingHotSpotAnalysis(os.path.join(home_dir, 'stc_4W.nc'), "FAELLEEWZ_7_MEAN_ZEROS", os.path.join(results_dir, 'emerg_4W'))


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um im ArcGIS-Enterprise ein Layer erzeugen zu können.

# In[139]:


emerg_4W_SDF = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'emerg_4W'))


# Um die Daten darstellen zu können muss ein Layer in dem Projektordner erzeugt werden. Damit nicht zu viele Layer mit der Zeit angelegt werden wird ein möglicher zuvor erstellter Layer gelöscht.

# In[140]:


gis = GIS("home")
items = gis.content.search(query ='emerg_4W')
for item in items:
    item.delete()
emerg_4W = emerg_4W_SDF.spatial.to_featurelayer('emerg_4W', tags=['Corona', 'Covid-19'], folder='Masterprojekt')


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[141]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
map_emerg_4W = gis.map("Germany")
classed_color_renderer =  {"renderer":"ClassedColorRenderer", "field_name":"category" }
map_emerg_4W.add_layer(emerg_4W, classed_color_renderer)
map_emerg_4W


# #### Visualisierung in 3D <a class="anchor" id="analyse-stc-vis3D"></a>
# 
# Die erstellten Space-Time Cubes können auch in einer 3-dimensionalen Karte angezeigt werden.
# 
# Der folgende Link erläutert das Vorgehen näher:
# https://pro.arcgis.com/en/pro-app/latest/tool-reference/space-time-pattern-mining/visualizecube3d.htm

# Für die Visualisierung wird eine Farbpalette benötigt.

# In[142]:


palette = ['blue', 'cornflowerblue', 'lightskyblue', 'white', 'salmon', 'tomato', 'red']
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['blue', 'cornflowerblue', 'lightskyblue', 'white', 'salmon', 'tomato', 'red'])


# ##### kleine Datenmenge (10.12.2020 - 24.12.2020)

# Das dreidimonsionale Rendering wirde über die Funktion **arcpy.stpm.VisualizeSpaceTimeCube3D()** durchgeführt. Dabei werden die einzelnen Bin als Ergebnis einer Hot Spot-Analyse dargestellt. Das Ergebnis wird in der Geodatabase abgelegt.

# In[143]:


arcpy.stpm.VisualizeSpaceTimeCube3D(os.path.join(home_dir, 'stc_Test.nc'), "FAELLEEWZ_7_NONE_ZEROS", "HOT_AND_COLD_SPOT_RESULTS", os.path.join(results_dir,'vis3D_Test'))


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um im ArcGIS-Enterprise ein Layer erzeugen zu können.

# In[144]:


vis3D_Test_SEDF = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'vis3D_Test'))
vis3D_Test_SEDF.head()


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[145]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
vis3D_Test_map = gis.map("Germany", mode='3D')
vis3D_Test_map.basemap = 'dark-gray-vector'
vis3D_Test_SEDF.spatial.plot(vis3D_Test_map, col="HS_BIN", cmap='bwr', renderer_type='c', method='esriClassifyNaturalBreaks', min_value=-3, class_count=7, alpha=0.8, line_width=0.2)
vis3D_Test_map


# ##### 1. Welle (02.03.2020 - 19.04.2020)

# Das dreidimonsionale Rendering wirde über die Funktion **arcpy.stpm.VisualizeSpaceTimeCube3D()** durchgeführt. Dabei werden die einzelnen Bin als Ergebnis einer Hot Spot-Analyse dargestellt. Das Ergebnis wird in der Geodatabase abgelegt.

# In[146]:


arcpy.stpm.VisualizeSpaceTimeCube3D(os.path.join(home_dir, 'stc_1W.nc'), "FAELLEEWZ_7_MEAN_ZEROS", "HOT_AND_COLD_SPOT_RESULTS", os.path.join(results_dir,'vis3D_1W'))


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um im ArcGIS-Enterprise ein Layer erzeugen zu können.

# In[147]:


vis3D_1W_SEDF = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'vis3D_1W'))
vis3D_1W_SEDF.head()


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[148]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
vis3D_1W_map = gis.map("Germany", mode='3D')
vis3D_1W_map.basemap = 'dark-gray-vector'
vis3D_1W_SEDF.spatial.plot(vis3D_1W_map, col="HS_BIN", cmap='bwr', renderer_type='c', method='esriClassifyNaturalBreaks', min_value=-3, class_count=7, alpha=0.8, line_width=0.2)
vis3D_1W_map


# ##### 2. Welle (05.10.2020 - 31.01.2021)

# Das dreidimonsionale Rendering wirde über die Funktion **arcpy.stpm.VisualizeSpaceTimeCube3D()** durchgeführt. Dabei werden die einzelnen Bin als Ergebnis einer Hot Spot-Analyse dargestellt. Das Ergebnis wird in der Geodatabase abgelegt.

# In[149]:


arcpy.stpm.VisualizeSpaceTimeCube3D(os.path.join(home_dir, 'stc_2W.nc'), "FAELLEEWZ_7_MEAN_ZEROS", "HOT_AND_COLD_SPOT_RESULTS", os.path.join(results_dir,'vis3D_2W'))


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um im ArcGIS-Enterprise ein Layer erzeugen zu können.

# In[150]:


vis3D_2W_SEDF = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'vis3D_2W'))
vis3D_2W_SEDF.head()


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[151]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
vis3D_2W_map = gis.map("Germany", mode='3D')
vis3D_2W_map.basemap = 'dark-gray-vector'
vis3D_2W_SEDF.spatial.plot(vis3D_2W_map, col="HS_BIN", cmap='bwr', renderer_type='c', method='esriClassifyNaturalBreaks', min_value=-3, class_count=7, alpha=0.8, line_width=0.2)
vis3D_2W_map


# ##### 3. Welle (01.03.2021 - 16.05.2021)

# Das dreidimonsionale Rendering wirde über die Funktion **arcpy.stpm.VisualizeSpaceTimeCube3D()** durchgeführt. Dabei werden die einzelnen Bin als Ergebnis einer Hot Spot-Analyse dargestellt. Das Ergebnis wird in der Geodatabase abgelegt.

# In[152]:


arcpy.stpm.VisualizeSpaceTimeCube3D(os.path.join(home_dir, 'stc_3W.nc'), "FAELLEEWZ_7_MEAN_ZEROS", "HOT_AND_COLD_SPOT_RESULTS", os.path.join(results_dir,'vis3D_3W'))


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um im ArcGIS-Enterprise ein Layer erzeugen zu können.

# In[153]:


vis3D_3W_SEDF = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'vis3D_3W'))
vis3D_3W_SEDF.head()


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[154]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
vis3D_3W_map = gis.map("Germany", mode='3D')
vis3D_3W_map.basemap = 'dark-gray-vector'
vis3D_3W_SEDF.spatial.plot(vis3D_3W_map, col="HS_BIN", cmap='bwr', renderer_type='c', method='esriClassifyNaturalBreaks', min_value=-3, class_count=7, alpha=0.8, line_width=0.2)
vis3D_3W_map


# ##### 4. Welle (04.10.2021 - 02.01.2022)

# Das dreidimonsionale Rendering wirde über die Funktion **arcpy.stpm.VisualizeSpaceTimeCube3D()** durchgeführt. Dabei werden die einzelnen Bin als Ergebnis einer Hot Spot-Analyse dargestellt. Das Ergebnis wird in der Geodatabase abgelegt.

# In[155]:


arcpy.stpm.VisualizeSpaceTimeCube3D(os.path.join(home_dir, 'stc_4W.nc'), "FAELLEEWZ_7_MEAN_ZEROS", "HOT_AND_COLD_SPOT_RESULTS", os.path.join(results_dir,'vis3D_4W'))


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um im ArcGIS-Enterprise ein Layer erzeugen zu können.

# In[156]:


vis3D_4W_SEDF = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'vis3D_4W'))
vis3D_4W_SEDF.head()


# Dieser Layer kann nun in einer Karte angezeigt werden.

# In[157]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
vis3D_4W_map = gis.map("Germany", mode='3D')
vis3D_4W_map.basemap = 'dark-gray-vector'
vis3D_4W_SEDF.spatial.plot(vis3D_4W_map, col="HS_BIN", cmap='bwr', renderer_type='c', method='esriClassifyNaturalBreaks', min_value=-3, class_count=7, alpha=0.8, line_width=0.2)
vis3D_4W_map


# #### Time Series Clustering <a class="anchor" id="analyse-stc-clust"></a>
# 
# Mit Hilfe der zuvor erstellten Space-Time Cubes sollen ähnliche Verläufe innerhalb der Wellen über ein Space Time Clustering zusammengefasst werden.
# 
# Die Funktionsweise des Werkzeugs kann hier nachvollzogen werden:
# https://pro.arcgis.com/en/pro-app/latest/tool-reference/space-time-pattern-mining/time-series-clustering.htm
# 
# Bei jedem Ausführen des Werkzeugs wird ein anderes Ergebnis berechnet. Dies geschieht aufgrund der zufällig gewählten Startpunkte beim Clustering.

# ##### kleine Datenmenge (10.12.2020 - 24.12.2020)

# Falls das Ergebnis des zu erstellenden Time Series Clustering aus vorherigen Durchläufen bereits vorliegt wird dieses gelöscht.
# 
# Das Time Series Clustering wird über die arcpy Funktion **arcpy.stpm.TimeSeriesClustering()** mit Hilfe der Space-Time Cubes berechnet und in der Geodatabase abgelegt.

# In[158]:


arcpy.management.Delete(os.path.join(results_dir, 'clust_test'))
arcpy.stpm.TimeSeriesClustering(os.path.join(home_dir, 'stc_Test.nc'), "FAELLEEWZ_7_NONE_ZEROS", os.path.join(results_dir,'clust_test'), "VALUE", '', '', '', 'CREATE_POPUP')


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um es anzeigen lassen zu können.

# In[159]:


clust_Test_sedf = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'clust_test'))
clust_Test_sedf


# Dieser Layer kann nun angezeigt in einer Karte werden.
# 
# Bei der Ausführung kam es bei uns zu einem serverseitigen Anzeigefehler. Daher kann unter [Datenexport](#export) die Geodatabase mit den Ergebnissen der Analysen exportiert werden und das Ergebnnis desktopseitig mit Hilfe von ArcGIS Pro angezeigt werden.

# In[ ]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
clust_Test_map = gis.map("Germany")
#clust_Test_sedf.spatial.plot(clust_Test_map, col="CLUSTER_ID", renderer_type='u', cmap="Set3")
clust_Test_sedf.spatial.plot(clust_Test_map)
clust_Test_map


# ##### 1. Welle (02.03.2020 - 19.04.2020)

# Falls das Ergebnis des zu erstellenden Time Series Clustering aus vorherigen Durchläufen bereits vorliegt wird dieses gelöscht.
# 
# Das Time Series Clustering wird über die arcpy Funktion **arcpy.stpm.TimeSeriesClustering()** mit Hilfe der Space-Time Cubes berechnet und in der Geodatabase abgelegt.

# In[161]:


arcpy.management.Delete(os.path.join(results_dir, 'clust_1W'))
arcpy.stpm.TimeSeriesClustering(os.path.join(home_dir, 'stc_1W.nc'), "FAELLEEWZ_7_MEAN_ZEROS", os.path.join(results_dir,'clust_1W'), "VALUE", '', '', '', 'CREATE_POPUP')


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um es anzeigen lassen zu können.

# In[162]:


clust_1W_sedf = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'clust_1W'))
clust_1W_sedf


# Dieser Layer kann nun angezeigt in einer Karte werden.
# 
# Bei der Ausführung kam es bei uns zu einem serverseitigen Anzeigefehler. Daher kann unter [Datenexport](#export) die Geodatabase mit den Ergebnissen der Analysen exportiert werden und das Ergebnnis desktopseitig mit Hilfe von ArcGIS Pro angezeigt werden.

# In[ ]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
clust_1W_map = gis.map("Germany")
#clust_1W_sedf.spatial.plot(clust_1W_map, col="CLUSTER_ID", renderer_type='u', cmap="Set3")
clust_1W_sedf.spatial.plot(clust_1W_map)
clust_1W_map


# ##### 2. Welle (05.10.2020 - 31.01.2021)

# Falls das Ergebnis des zu erstellenden Time Series Clustering aus vorherigen Durchläufen bereits vorliegt wird dieses gelöscht.
# 
# Das Time Series Clustering wird über die arcpy Funktion **arcpy.stpm.TimeSeriesClustering()** mit Hilfe der Space-Time Cubes berechnet und in der Geodatabase abgelegt.

# In[163]:


arcpy.management.Delete(os.path.join(results_dir, 'clust_2W'))
arcpy.stpm.TimeSeriesClustering(os.path.join(home_dir, 'stc_2W.nc'), "FAELLEEWZ_7_MEAN_ZEROS", os.path.join(results_dir,'clust_2W'), "VALUE", '', '', '', 'CREATE_POPUP')


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um es anzeigen lassen zu können.

# In[164]:


clust_2W_sedf = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'clust_2W'))
clust_2W_sedf


# Dieser Layer kann nun angezeigt in einer Karte werden.
# 
# Bei der Ausführung kam es bei uns zu einem serverseitigen Anzeigefehler. Daher kann unter [Datenexport](#export) die Geodatabase mit den Ergebnissen der Analysen exportiert werden und das Ergebnnis desktopseitig mit Hilfe von ArcGIS Pro angezeigt werden.

# In[ ]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
clust_2W_map = gis.map("Germany")
#clust_2W_sedf.spatial.plot(clust_2W_map, col="CLUSTER_ID", renderer_type='u', cmap="Set3")
clust_2W_sedf.spatial.plot(clust_2W_map)
clust_2W_map


# ##### 3. Welle (01.03.2021 - 16.05.2021)

# Falls das Ergebnis des zu erstellenden Time Series Clustering aus vorherigen Durchläufen bereits vorliegt wird dieses gelöscht.
# 
# Das Time Series Clustering wird über die arcpy Funktion **arcpy.stpm.TimeSeriesClustering()** mit Hilfe der Space-Time Cubes berechnet und in der Geodatabase abgelegt.

# In[165]:


arcpy.management.Delete(os.path.join(results_dir, 'clust_3W'))
arcpy.stpm.TimeSeriesClustering(os.path.join(home_dir, 'stc_3W.nc'), "FAELLEEWZ_7_MEAN_ZEROS", os.path.join(results_dir,'clust_3W'), "VALUE", '', '', '', 'CREATE_POPUP')


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um es anzeigen lassen zu können.

# In[166]:


clust_3W_sedf = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'clust_3W'))
clust_3W_sedf


# Dieser Layer kann nun angezeigt in einer Karte werden.
# 
# Bei der Ausführung kam es bei uns zu einem serverseitigen Anzeigefehler. Daher kann unter [Datenexport](#export) die Geodatabase mit den Ergebnissen der Analysen exportiert werden und das Ergebnnis desktopseitig mit Hilfe von ArcGIS Pro angezeigt werden.

# In[ ]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
clust_3W_map = gis.map("Germany")
#clust_3W_sedf.spatial.plot(clust_3W_map, col="CLUSTER_ID", renderer_type='u', cmap="Set3")
clust_3W_sedf.spatial.plot(clust_3W_map)
clust_3W_map


# ##### 4. Welle (04.10.2021 - 02.01.2022)

# Falls das Ergebnis des zu erstellenden Time Series Clustering aus vorherigen Durchläufen bereits vorliegt wird dieses gelöscht.
# 
# Das Time Series Clustering wird über die arcpy Funktion **arcpy.stpm.TimeSeriesClustering()** mit Hilfe der Space-Time Cubes berechnet und in der Geodatabase abgelegt.

# In[167]:


arcpy.management.Delete(os.path.join(results_dir, 'clust_4W'))
arcpy.stpm.TimeSeriesClustering(os.path.join(home_dir, 'stc_4W.nc'), "FAELLEEWZ_7_MEAN_ZEROS", os.path.join(results_dir,'clust_4W'), "VALUE", '', '', '', 'CREATE_POPUP')


# Das Ergebnis der Analyse wird in ein "spatially-enabled" Dataframe umgewandelt um es anzeigen lassen zu können.

# In[168]:


clust_4W_sedf = pd.DataFrame.spatial.from_featureclass(os.path.join(results_dir,'clust_4W'))
clust_4W_sedf


# Dieser Layer kann nun angezeigt in einer Karte werden.
# 
# Bei der Ausführung kam es bei uns zu einem serverseitigen Anzeigefehler. Daher kann unter [Datenexport](#export) die Geodatabase mit den Ergebnissen der Analysen exportiert werden und das Ergebnnis desktopseitig mit Hilfe von ArcGIS Pro angezeigt werden.

# In[ ]:


gis = GIS(url="https://arcgis.services.fbbgg.hs-woe.de/arcgis")
clust_4W_map = gis.map("Germany")
#clust_4W_sedf.spatial.plot(clust_4W_map, col="CLUSTER_ID", renderer_type='u', cmap="Set3")
clust_4W_sedf.spatial.plot(clust_4W_map)
clust_4W_map


# ## Datenexport <a class="anchor" id="export"></a>

# Über den folgenden Ablauf kann die Geodatabase im home-Verzeichnis des ArcGIS-Enterprise heruntergeladen werden. Diese kann entpackt und in ArcGIS Pro eingeladen werden. Damit können alle darin enthaltenen Ergebnisse angezeigt werden.

# In[169]:


import zipfile
zip = zipfile.ZipFile(os.path.join(home_dir, 'Results.zip'), 'w', zipfile.ZIP_DEFLATED)
path = os.path.normpath(results_dir)
for (dirpath, dirnames, filenames) in os.walk(path):
    for file in filenames:
        zip.write(os.path.join(dirpath, file),
        os.path.join(os.path.basename(path), os.path.join(dirpath, file)[len(path)+len(os.sep):]))
zip.close()

