# Tehtävä 1 #
* Pythonissa paketti managereja ovat esimerkiksi pip, anaconda ja virtualENV

# Tehtävä 2#
* Oman paketin lisääminen pip:in käyttämään PyPi kirjastoon on helppoa ja sen voi tehdä kuka vaan netistä helposti löytyvillä ohjeilla.

# Tehtävä 3#
* Löysin ainakin yhden hyökkäyksen, josta raportoidaan [tässä](https://cyber-reports.com/2021/02/20/dependency-confusion-attack-mounted-via-pypi-repo-exposes-flawed-package-installer-behavior/)

# Tehtävä 4 #

    - Hyökkääjä lisää paketin PyPi
            - Käyttäjä asentaa paketin
                -Käyttäjä käyttää pakettia ohjelmassa
                    - Paketin haittakoodi ajetaan
                        - Kehittäjän ohjelma lähettää tietoa hyökkääjälle
                    - Paketin koodi ajetaan asennuksen yhteydessä
                        - Kehittäjän tietokone lähettää tietoa hyökkääjälle
            - Haittapaketti huomataan
                 - Haittapaketti poistetaan 
  
# Tehtävä 5 #               
* Jos paketteja asennettaessa pip ei kerrota lokaalin paketin sijaintia, se koittaa asentaa paketin PyPi kirjastosta
* pip install -e c:\users\worker\src\test\lib\esimerkkiPaketti
* saadaan asennettua lokaalipaketti