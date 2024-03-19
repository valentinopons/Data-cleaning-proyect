# -*- coding: utf-8 -*-
"""
Creado: 16/02/2024
Materia: Laboratorio de Datos - FCEyN - UBA
@autores:  Pons Valentino, Porcel Carlos, Suarez Javier
"""
### Importo librearias
import pandas as pd
import numpy as np
from inline_sql import sql
import matplotlib.pyplot as plt
from matplotlib import ticker, rcParams
import seaborn as sns

### cargo los datos crudos
  
# cargo los dataframes de TWB

pbi_paises = pd.read_csv('./TablasOriginales/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_73.csv',skiprows=4)

datos_paises = pd.read_csv('./TablasOriginales/Metadata_Country_API_NY.GDP.PCAP.CD_DS2_en_csv_v2_73.csv')

metadata = pd.read_csv('./TablasOriginales/Metadata_Indicator_API_NY.GDP.PCAP.CD_DS2_en_csv_v2_73.csv')

# cargo Dataframes del Ministerio

secciones_original = pd.read_csv('./TablasOriginales/lista-secciones.csv')

sedes_original = pd.read_csv('./TablasOriginales/lista-sedes.csv')

datos_sedes = pd.read_csv('./TablasOriginales/lista-sedes-datos.csv')
    
### Creo los esquemas necesarios para el proyecto 

# creo los esquemas en dataframe vacios

redes_sociales = pd.DataFrame(columns=['id_sede','url'])

sedes = pd.DataFrame(columns=['id_sede','nom_sede','codigo_pais'])

secciones  = pd.DataFrame(columns=['id_sede','nom_seccion'])

paises  = pd.DataFrame(columns=['codigo_pais','nom_pais','id_region','pbi'])

regiones = pd.DataFrame(columns=['id_region','nom_region'])



### importo los datos a los dataframe vacios

# Regiones
# importo:
#   nom_region del dataframe datos_paises
regiones['nom_region'] = datos_sedes['region_geografica'].sort_values().unique() # selecciones los valores, ordeno y luego dejo solo los valores unicos
#   genero un id_regino para cada region
regiones['id_region'] = np.array(range(1,len(regiones['nom_region'])+1))# el array es  [1,2,3,4,5,6,7,8,9]  

# Paises
# importo:
#   codigo_pais, nom_pais, id_region del dataframe datos_paises
paises[['codigo_pais','nom_pais','id_region']] = datos_sedes[['pais_iso_3','pais_castellano','region_geografica']]
#   estandarizo GRB a GBR 
paises["codigo_pais"] = paises["codigo_pais"].replace(['GRB'],'GBR')
#   pib del dataframe pib_paises
paises['pbi'] = pd.merge(paises['codigo_pais'], pbi_paises[['Country Code','2022']],
                         left_on='codigo_pais', right_on='Country Code',how='left')['2022']# Hago un natural join con codigo_pais del dataframe paises
                                                                                # y Country Code de pbi_paises para accedes a el pbi en 2022,
                                                                                # esos valores los importo a la columna pbi
# Sedes
# importo:
#   id_sede,nom_sede,codigo_territorio del dataframe datos_sedes
sedes[['id_sede','nom_sede','codigo_pais']] = datos_sedes[['sede_id','sede_desc_castellano','pais_iso_3']]

# Secciones
# importo:
#   id_sede, nom_seccion del dataframe secciones_original
secciones[['id_sede','nom_seccion']] = secciones_original[['sede_id','sede_desc_castellano']]

# Redes Sociales
#   importo id_sede, url del dataframe datos_sedes
redes_sociales[['id_sede','url']] = datos_sedes[['sede_id','redes_sociales']]



### Limpieza y estandarizacion

# Paises
#   elimino los duplicados
paises.drop_duplicates(inplace=True)
#   elimino las regiones de mi dataframe paises
paises.dropna(inplace=True)
#paises[paises['id_region'].notnull()]
#   cambio las regiones por su id
paises['id_region'] = paises['id_region'].apply(lambda x: regiones[regiones.loc[:,'nom_region'] == x]['id_region'].values[0])
#   reseto sus indices
paises.reset_index(inplace=True,drop=True)

# Sedes
#   estandarizo GRB a GBR
sedes["codigo_pais"] = sedes["codigo_pais"].replace(['GRB'],'GBR')

# Redes Sociales:   
#   Crep un dataframe con los datos sin precesar
redes_sociales_s_procesar = redes_sociales.copy()
#   Vacion el dataframe   
redes_sociales = pd.DataFrame(columns=['id_sede','url'])

#   Elimino las sedes que no tienen redes sociales
redes_sociales_s_procesar.dropna(inplace=True)
#   Reseteo el indice
redes_sociales_s_procesar.reset_index(inplace=True,drop=True)
#   Vuelvo los valores de la columna url en atomicos
for i in range(redes_sociales_s_procesar.shape[0]):
    # accedo al valor de la columna redes_sociales por la posicion en i
    urls =  redes_sociales_s_procesar.loc[i,'url']   
    # creo una lista con las urls eliminando el ultimo elemento
    lista_urls = urls.split('  //  ')[:-1]
    # si la lista tiene mas de un elemento significa que elutlimo elemento es un espacio, por lo cual lo quito
    # creo una lista de igual longitud que lista_urls con todos sus valores igual a la sede que pertenece las urls
    lista_full_sede_id = np.full(len(lista_urls), redes_sociales_s_procesar.loc[i,'id_sede'])
    # creo una lista de tuplas (sede_id, urls)
    lista_tuplas_urls = zip(lista_full_sede_id,lista_urls)
    # genero un dataframe con las tuplas de lista_tuplas_urls
    df_url_sede = pd.DataFrame(lista_tuplas_urls,columns=['id_sede','url'])
    # concateno las las nuevas urls al dataframe de redes sociales
    redes_sociales = pd.concat([redes_sociales,df_url_sede], axis=0,ignore_index=True)  

#Elimino urls considerados de mala calidad (fijarse en el informe en toma de decicsiones que son considerados url de mala calidad)
redes_sociales = sql^ """
         SELECT *
         FROM redes_sociales
         WHERE url LIKE 'http%' OR url LIKE 'www%' OR url LIKE 'twitter%' OR url LIKE 'Twitter%'
                                OR url LIKE 'facebook%' OR url LIKE 'Facebook%' OR url LIKE 'instagram%'
                                OR url LIKE 'Instagram%' OR url LIKE 'linkedin%'
        
                """       
#   reseteo el indice de redes_sociales
redes_sociales.reset_index(inplace=True,drop=True)

# Secciones
#   Elimino duplicados
secciones.drop_duplicates(inplace=True)



### Expoerto los dataframes
lista_esquemas = [regiones,paises,sedes,secciones,redes_sociales]
nombre_cvs  = ['regiones.csv','paises.csv','sedes.csv','secciones.csv','redes_sociales.csv']
for esquema,nombre in zip(lista_esquemas,nombre_cvs):
    esquema.to_csv('./TablasLimpias/'+nombre,index=False, encoding = 'utf-8')


# =============================================================================
#  h)     
# =============================================================================
# i) Para resolver este punto, se eliminaron 3 filas de la tabla de secciones_original
# ya que habia sedes que tenian 2 secciones con el mismo nombre y lo que sucedia era que
# esas secciones tenian distinto jefe, entonces las contaba como 2. 
# Nosotros decidimos que eso solo cuente como uno. En este caso la modificacion a nivel tabla 
# seria la eliminacion de apenas 3 filas de las 516 que lleva la tabla secciones.
# La modificacion se llevo a cabo en la creacion del DataFrame de secciones (linea 295)
# Luego de creada la tabla, se le aplico un Select Distinct* y se lo igualo a secciones. 

cantidad_sedes = sql^"""
SELECT paises.nom_pais, count() AS sedes
FROM paises
LEFT OUTER JOIN sedes
ON paises.codigo_pais = sedes.codigo_pais
GROUP BY paises.nom_pais
 """

paises_sedes_secciones = sql^"""
SELECT paises.nom_pais, paises.codigo_pais, 
sedes.id_sede, secciones.nom_seccion
FROM paises
LEFT OUTER JOIN sedes 
ON paises.codigo_pais = sedes.codigo_pais
LEFT OUTER JOIN secciones
ON sedes.id_sede = secciones.id_sede
"""

cantidad_secciones = sql^"""
SELECT paises_sed_sec.nom_pais, paises_sed_sec.codigo_pais, 
paises_sed_sec.id_sede,
SUM(CASE WHEN paises_sed_sec.nom_seccion IS NULL THEN 0 ELSE 1 END ) as cantidad_secciones
FROM paises_sedes_secciones AS paises_sed_sec
GROUP BY paises_sed_sec.nom_pais, paises_sed_sec.codigo_pais, 
paises_sed_sec.id_sede
"""

promedio_seccionesxsede = sql^"""
SELECT cantidad_secciones.nom_pais, 
cantidad_secciones.codigo_pais, 
ROUND(AVG(cantidad_secciones.cantidad_secciones), 1) as secciones_promedio
FROM cantidad_secciones
GROUP BY cantidad_secciones.nom_pais, cantidad_secciones.codigo_pais
"""

ejercicio1 = sql^"""
SELECT paises.nom_pais AS Pais, cantidad_sedes.sedes,
promedio_seccionesxsede.secciones_promedio,
paises.pbi AS Pbi_per_Capita_2022_U$S
FROM paises
LEFT OUTER JOIN cantidad_sedes 
ON cantidad_sedes.nom_pais = paises.nom_pais
LEFT OUTER JOIN promedio_seccionesxsede
ON promedio_seccionesxsede.nom_pais = paises.nom_pais
ORDER BY cantidad_sedes.sedes DESC, paises.nom_pais 
"""
ejercicio1= sql^"""
SELECT *
FROM ejercicio1
WHERE Pbi_per_Capita_2022_U$S IS NOT NULL
"""

# Conclusion de lo observado: En principio, se observan resultados muy disparejos si se
# intentan relacionar los datos de sedes, secciones y pbi.
# Sin embargo, pareciera ser que la mayoria de los paises de la tabla 
# que mas PBI poseen son aquellos que poseen secciones en promedio mayores o iguales a 2.5 


# ii) La solucion final a este ejercicio 
# es la tabla de consulta "ejercicio2" 

promedio_pbi = sql^"""
SELECT regiones.nom_region AS Region_geografica, ejercicio1.*
FROM ejercicio1
LEFT OUTER JOIN paises
ON ejercicio1.Pais = paises.nom_pais
LEFT OUTER JOIN regiones
ON regiones.id_region = paises.id_region
"""
ejercicio2 = sql^"""
SELECT promedio_pbi.Region_geografica, 
COUNT(*) AS Paises_Con_Sedes_Argentinas,
ROUND(AVG(promedio_pbi.Pbi_per_Capita_2022_U$S),2) AS Promedio_Pbi_per_Capita_2022_U$S
FROM promedio_pbi
GROUP BY promedio_pbi.Region_geografica
ORDER BY promedio_Pbi_per_Capita_2022_U$S DESC
"""

#ejercicio IV)
"""
Genero un nuevo data frame llamado reporte el cual es igual a redes_sociales 

"""
reporte = sql^ """
         SELECT id_sede AS sede_id  , url
         FROM redes_sociales
         """
reporte = sql^ """
          SELECT sede_id ,
          CASE 
              WHEN url LIKE '%twitter%' OR url LIKE '%Twitter%' THEN 'Twitter'
              WHEN url LIKE '%instagram%' OR url LIKE '%Instagram%' THEN 'Instagram'
              WHEN url LIKE '%facebook%' OR url LIKE '%Facebook%' THEN 'Facebook'
              WHEN url LIKE '%linkedin%' OR url LIKE '%Linkedin%' THEN 'Linkedin'
              WHEN url LIKE '%flickr%' OR url LIKE '%Flickr%' THEN 'Flickr'
              WHEN url LIKE '%youtube%' OR url LIKE '%Youtube%' THEN 'Youtube'
              ELSE ''
              END AS Red_social , url
             FROM reporte 
                """
reporte = sql^ """
            SELECT pais_castellano AS Pais , r.sede_id AS Sede , Red_social , url AS URL
            FROM reporte AS r
            LEFT JOIN datos_sedes AS d
            ON r.sede_id = d.sede_id
                """
#Ordeno                
reporte = sql^ """  
            SELECT *
            FROM reporte
            ORDER BY Pais ASC , Sede ASC , Red_social ASC , URL ASC
                """

#ejercicio III)
"""
IMPORTANTE: Para este ejercicio uso dataframe "reporte" generado en el ejercicio 4. 

"""
res = sql^"""
            SELECT Pais , Red_social , COUNT(Red_social) AS nro_redes
            FROM reporte
            GROUP BY Pais , Red_social
            """
tabla_h_III = sql^"""
        SELECT Pais , COUNT(Pais) AS nro_redes
        FROM res
        GROUP BY Pais
        ORDER BY Pais ASC , nro_redes ASC

        """





#exporto archivos .csv
lista_consultas = [ejercicio1, ejercicio2,tabla_h_III,reporte]
nombre_cvs_sql  = ['ejercico1.csv','ejercico2.csv','ejercico3.csv','ejercico4.csv']
for esquema,nombre in zip(lista_consultas,nombre_cvs_sql):
    esquema.to_csv('./TablasConsultas/'+nombre,index=False, encoding = 'utf-8')

# =============================================================================
# i) Mostrar, utilizando herramientas de visualizacion, la siguiente informacion: 
# 
# i) Mostrar cantidad de sedes por region geografica. 
#    Ordenados de manera decreciente por dicha cantidad.
# =============================================================================

sedesXregion = sql^"""
SELECT Region_geografica, SUM(sedes) as Cantidad_Sedes
FROM promedio_pbi
GROUP BY Region_geografica
"""

#Dataframe de ejercicio2 modificado para que aparesca ordenado en el grafico
sedesXregionOrd = sedesXregion.sort_values('Cantidad_Sedes', ascending=False)

#lista de 9 colores para las regiones
colores = ['blue', 'green', 'purple', 'blue', 'blue', 'orange', 'green', 'orange', 'gray']

#Crea el grafico de barras
bars = plt.bar(sedesXregionOrd['Region_geografica'],
        sedesXregionOrd['Cantidad_Sedes'],
        color=colores)

#Linea punteada de cada valor en y 
plt.grid(axis='y', color='gray', linestyle='dashed') 

#Rotar los textos en el eje x para que sean legibles
plt.xticks(rotation=60, ha='right', fontsize=8)  # 'ha' es para alineación horizontal

#Aca se le agrega la cantidad arriba de cada bin
plt.bar_label(bars, fmt='%d', fontsize=9, label_type='edge', color='black', weight='bold')

#Agregar titulo y limites al eje y
plt.title('Cantidad de Sedes por Región')
plt.ylim(0,50)
plt.gca().spines['left'].set_visible(False) # se borra linea derecha de margen
plt.gca().spines['top'].set_visible(False) # se borra linea debajo del titulo
plt.tick_params(axis='y', left=False) # se borra tick de valores del eje y
plt.gca().set_facecolor('#D7DBDD') # Agregar fondo de color al gráfico
plt.gcf().set_facecolor('#D7DBDD') # Agregar fondo de color al gráfico


# bars = plt.bar(ejercicio2ord['Region_geografica'], 
#                ejercicio2ord['Paises_Con_Sedes_Argentinas'], 
#                color='black')


# =============================================================================
# Conclusion: 
# Como era de esperarse, nuestro continente americano es el que tiene
# mayor cantidad de sedes argentinas, que cuenta con 75 sedes.
# En segundo lugar tenemos a Europa con 43 sedes y al parecer se le da primero 
# mayor relevancia diplomatica al sector de Europa Occidental 
# que a Europa Central y Oriental con una aplica diferencia de sedes.
# El continente africano cuenta con muy pocas sedes argentinas, pero no tanto como
# Oceania que cuenta con solo 3 sedes argentinas. 
# Seria interesante que Argentina lograra mayor contacto con el continente de Oceania 
# =============================================================================


#EJERCICIO i II
#Grafico de biotes del PBI por Region donde Argentina tiene sedes

#creo un dataframe con las regiones y sus pbi
paises_regiones = pd.merge(paises,regiones)[['nom_region','pbi']]

fig,ax = plt.subplots(figsize=(30,15))

rcParams['font.family'] = 'sans-serif'           # Modifica el tipo de letra
rcParams['axes.spines.right']  = False            # Elimina linea derecha   del recuadro
rcParams['axes.spines.left']   = True             # Agrega  linea izquierda del recuadro
rcParams['axes.spines.top']    = False            # Elimina linea superior  del recuadro
rcParams['axes.spines.bottom'] = False             # Elimina linea inferior  del recuadro

#creo un array para ordenar por mediana por region de forma ascendente
indice_ord = paises_regiones.groupby('nom_region').median().sort_values('pbi').index

ax = sns.boxplot(x='nom_region',
            y='pbi',
            data= paises_regiones,
            order=indice_ord,
            palette=['orange','orange','blue','purple','blue','green','green','blue','gray']
            )

plt.title('Gráfico de caja PBI por Region') 

ax.set_xticklabels(indice_ord,fontsize = 17)        # aumento tamño etiquetas del eje x
ax.set_xlabel('Regiones',fontsize=25)               # Coloco titulo con tamaño 25 al eje x
plt.xticks(rotation=65)                             # Roto las etiqueta del eje x

ax.set_yticklabels(ax.get_yticks(),fontsize = 15)   # aumento tamño etiquetas del eje y
ax.set_ylabel('PBI per capita (USD)',fontsize=25)   # Coloco titulo con tamaño 25 al eje y


plt.show()


#EJERCICIO i III
#Grafico PBI y numero de sedes argentinas por pais.

pbi_2022_sedes = sql^"""
                    SELECT paises.nom_pais , pbi , sedes
                    FROM paises
                    LEFT JOIN cantidad_sedes
                    ON paises.nom_pais = cantidad_sedes.nom_pais
                    ORDER BY pbi ASC
                    """ 
                    
y = pbi_2022_sedes['pbi']
x = pbi_2022_sedes['sedes'] 
plt.xticks(range(0, 13, 1))
plt.yticks(range(0,110000,20000))
rcParams['font.family'] = 'sans-serif'
plt.title('Relacion PBI y numero de sedes argentinas por pais')  
           # Modifica el tipo de letra
rcParams['axes.spines.right'] = False            
rcParams['axes.spines.left']  = True            
rcParams['axes.spines.top']   = False  
plt.ylabel('PBI (usd)')
plt.xlabel('nroº de sedes')
plt.scatter(x,y,s=5 )
























