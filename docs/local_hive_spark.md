# Local Hive with PySpark Setup

This guide walks you through installing and configuring a fully‑functional, local Hive and PySpark development environment on Apple Silicon (M1/M2/M3) macOS machines. We install Java 8–dependent Hive alongside Hadoop, configure HDFS and YARN, set up MySQL as Hive’s metastore, and integrate Hive with Spark so you can run PySpark queries against your Hive warehouse.

Follow each section—from bootstrapping Homebrew and formatting HDFS to tuning `spark-defaults.conf`—to get Hadoop, Hive, and PySpark all working seamlessly on your Mac.

## Important Notes

**For both those configuration files and instructions here, replace `$HOME` with the path of your home folder.**

These instructions largely work for Linux too, just be aware of macOS specific commands and swap accordingly.

## Versions

| Hadoop | Hive   | Spark  | macOS  | Ubuntu |
| ------ | ------ | ------ | ------ | ------ |
| 3.4.1  | 4.0.1  | 3.5.5  | 15.4   | 22.04  |

## Install Pre-requisites

Download from the following locations and expand into your home folder:

- [Hadoop](https://hadoop.apache.org/releases.html)
- [Hive](https://www.apache.org/dyn/closer.cgi/hive/)
- [Spark](https://spark.apache.org/downloads.html)

Install Java 8 and 17:

- Java 8: `brew install --cask temurin@8`
- Java 17: `brew install openjdk@17`

---

## Environment variables

Add these lines to the end of `~/.zshrc`:

```sh
export HADOOP_HOME=$HOME/hadoop-3.4.1
export HIVE_HOME=$HOME/apache-hive-4.0.1-bin
export SPARK_HOME=$HOME/spark-3.5.5-bin-hadoop3
export PATH=$PATH:$HIVE_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
```

Note the version number and variant of the downloaded software and adjust accordingly.

---

## Hadoop Configuration

[Reference](https://medium.com/@daibinraju/installing-hadoop-with-hive-on-mac-m1-using-homebrew-3505c6166e83)

Edit `$HADOOP_HOME/etc/hadoop/hadoop-env.sh` to update JAVA_HOME:

`export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home`

Edit `$HADOOP_HOME/etc/hadoop/core-site.xml` to add the following configuration:

```xml
<configuration>
 <property>
  <name>fs.defaultFS</name>
  <value>hdfs://localhost:8020</value>
 </property>
</configuration>
```

Edit `$HADOOP_HOME/etc/hadoop/hdfs-site.xml` with:

```xml
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
  <property>
    <name>dfs.permissions</name>
    <value>false</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>file://$HOME/hdfs/namenode</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>file://$HOME/hdfs/datanode</value>
  </property>
</configuration>
```

Edit `$HADOOP_HOME/etc/hadoop/mapred-site.xml` with:

```xml
<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
  <property>
    <name>mapreduce.application.classpath</name>   
    <value>$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/*:$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/lib/*</value>
  </property>
</configuration>
```

Edit `$HADOOP_HOME/etc/hadoop/yarn-site.xml` with:

```xml
<configuration>

<!-- Site specific YARN configuration properties -->
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
  <property>
    <name>yarn.nodemanager.env-whitelist</name>  
    <value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PREPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_MAPRED_HOME</value>
  </property>
</configuration>
```

- Enable SSH
  - Create `~/.ssh/id_rsa` (if not already present) and run `cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys`. This allows local SSH logins without a password.
  - SSH login permissions is a must: `sudo systemsetup -setremotelogin on`. Test with `ssh localhost` (<https://stackoverflow.com/a/42037840/2844684>)
  - Edit macOS `/etc/hosts` file to point name of localhost (e.g. `macbookhostname`) to `127.0.0.1`
  - Terminal and/or VS Code need Full Disk Access permissions to turn on SSH. Turn them on in System Settings → Privacy & Security → Full Disk Access.

- Format HDFS: `hadoop namenode -format`
  - After format, delete all files under each DataNode’s `dfs.datanode.data.dir`. (We only have one Datanode in this guide because it is local and not an actual cluster.)

- Run Hadoop

```sh
sudo systemsetup -setremotelogin on
start-all.sh
```

- Operation
  - Shutdown Hadoop with `stop-all.sh`
  - Dashboard: <http://localhost:9870>

### Troubleshooting Hadoop

- Logs are found in: `$HADOOP_HOME/logs`

See running Hadoop processes using `jps`. We want to see these (ignore the process IDs):

```txt
24182 SecondaryNameNode
23911 NameNode
24027 DataNode
24524 NodeManager
24687 Jps
24415 ResourceManager
```
  
---

## Hive Configuration

- Go to `$HIVE_HOME/conf`
- Copy `hive-default.xml.template` to `hive-site.xml`
- Edit properties in `hive-site.xml` as follows:

```xml
  <property>
    <name>hive.exec.local.scratchdir</name>
    <value>$HOME/hive_logs/scratchdir</value>
    <description>Local scratch space for Hive jobs</description>
  </property>
  <property>
    <name>hive.downloaded.resources.dir</name>
    <value>$HOME/hive_logs/${hive.session.id}_resources</value>
    <description>Temporary local directory for added resources in the remote file system.</description>
  </property>

  <property>
    <name>hive.metastore.db.type</name>
    <value>mysql</value>
    <description>
      Expects one of [derby, oracle, mysql, mssql, postgres].
      Type of database used by the metastore. Information schema &amp; JDBCStorageHandler depend on it.
    </description>
  </property>

  <property>
    <name>hive.metastore.warehouse.dir</name>
    <value>hdfs://localhost:8020/user/hive/warehouse</value>
    <description>location of default database for the warehouse</description>
  </property>

  <property>
    <name>javax.jdo.option.ConnectionPassword</name>
    <value>password123</value>
    <description>password to use against metastore database</description>
  </property>

  <property>
    <name>javax.jdo.option.ConnectionURL</name>
    <value>jdbc:mysql://localhost:3306/hive?createDatabaseIfNotExist=true</value>
    <description>
      JDBC connect string for a JDBC metastore.
      To use SSL to encrypt/authenticate the connection, provide database-specific SSL flag in the connection URL.
      For example, jdbc:postgresql://myhost/db?ssl=true for postgres database.
    </description>
  </property>

  <property>
    <name>datanucleus.schema.autoCreateTables</name>
    <value>true</value>
  </property>

  <property>
    <name>javax.jdo.option.ConnectionDriverName</name>
    <value>com.mysql.cj.jdbc.Driver</value>
    <description>Driver class name for a JDBC metastore</description>
  </property>

  <property>
    <name>javax.jdo.option.ConnectionUserName</name>
    <value>hive</value>
    <description>Username to use against metastore database</description>
  </property>

  <property>
    <name>hive.querylog.location</name>
    <value>$HOME/hive_logs</value>
    <description>Location of Hive run time structured log file</description>
  </property>

  <property>
    <name>hive.server2.logging.operation.log.location</name>
    <value>$HOME/hive_logs/operation_logs</value>
    <description>Top level directory where operation logs are stored if logging functionality is enabled</description>
  </property>

  <property>
    <name>hive.server2.enable.doAs</name>
    <value>false</value>
    <description>
      Setting this property to true will have HiveServer2 execute
      Hive operations as the user making the calls to it.
    </description>
  </property>

  <property>
    <name>hive.scheduled.queries.executor.enabled</name>
    <value>false</value>
    <description>Controls whether HS2 will run scheduled query executor.</description>
  </property>
```

### Hive Logs

Edit `hive-log4j2.properties`:

```conf
property.hive.log.dir = $HOME/hive_logs
```

### Setup MySQL for Hive Metastore

Go to <https://dev.mysql.com/downloads/connector/j/> select operating system as Platform Independent, and download and extract the tar file.

Copy the jar file into Hive's lib folder and spark's lib folder:

```sh
cp mysql-connector-j-8.0.31.jar $HIVE_HOME/lib/
cp mysql-connector-j-8.0.31.jar $SPARK_HOME/jars/
```

Note the jar file extracted above might be of a different version. Adjust accordingly.

If the above is not done, PySpark will have this error:

```txt
Attempt to invoke the "HikariCP" plugin to create a ConnectionPool gave an error : The specified datastore driver ("com.mysql.cj.jdbc.Driver") was not found in the CLASSPATH. Please check your CLASSPATH specification, and the name of the driver.
```

Install and run MySQL:

```sh
brew install mysql
mysql.server start
mysql -u root
```

After the MySQL service is up, log in to MySQL as root user, create a new user and grant privileges. The username is `hive` and password is `password123`.

After login, run the flowing commands:

```sql
create user 'hive'@'localhost' identified by 'password123';
GRANT ALL PRIVILEGES ON  *.* to 'hive'@'localhost';
FLUSH PRIVILEGES;
```

Initialize metastore schema:

```sh
schematool -initSchema -dbType mysql
```

- Hive is now setup to be the metastore for Spark.

---

## PySpark

Edit `$SPARK_HOME/conf/spark-defaults.conf` with:

```conf
spark.sql.catalogImplementation=hive
```

Symlink Hadoop and Hive configuration files into Spark's `$SPARK_HOME/conf` directory.

```sh
ln -s $HADOOP_HOME/etc/hadoop/hdfs-site.xml .
ln -s $HADOOP_HOME/etc/hadoop/core-site.xml .
ln -s $HIVE_HOME/conf/hive-site.xml .
```

### Example PySpark usage

We are done, PySpark now works with Hive as their metastore for SQL tables.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("test_pyspark") \
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.memory", "12g") \
    .config("spark.sql.orc.enableVectorizedReader", "false") \
    .config("spark.sql.parquet.columnarReaderBatchSize", "1024") \
    .config("spark.sql.orc.columnarReaderBatchSize", "1024") \
    .getOrCreate()

spark.sql("SHOW TABLES").show()
```

Tip: To avoid seeing many warnings from Hive, run:

```python
spark.sparkContext.setLogLevel("ERROR")
```

---

## References

- PySpark: <https://datacouch.medium.com/how-to-set-up-spark-environment-on-mac-c1553005e1f4>
- Hive: <https://medium.com/@daibinraju/installing-hadoop-with-hive-on-mac-m1-using-homebrew-3505c6166e83>
- Hadoop: <<https://medium.com/@MinatoNamikaze02/installing-hadoop-on-macos-m1-m2-2023-d963abeab38e>>

---

## Appendix

### Running PySpark without Hadoop

Without Hadoop and Hive, PySpark stores the metadata of tables in a derby database and the contents of the tables in the current working directory. They are stored in the folders `metastore_db` and `spark-warehouse` respectively.

For my workflow which concurrent notebooks and test runners, it's a blocker as the derby locks the whole database for one process. It's also not fun to manage the current working directory (so they point to the same folder) as my notebooks and tests are run from separate directories.
These are other advantages of setting Hadoop/Yarn locally for Spark

**Realistic environment**
You’re testing against the same services (HDFS, YARN, Hive metastore) you’d use in production—so you’ll catch config quirks, classpath issues, and permission problems before they hit the cluster.

**Distributed semantics (sort of)**
Even on one machine, the multi-process nature gives you a taste of data locality, block replication, shuffle behavior, and RPCs between components.

**Metastore durability & sharing**
A local MySQL metastore persists metadata across sessions and can, in principle, be accessed concurrently by multiple Spark/Hive clients in your dev environment.

**End-to-end testing**
You can validate end-to-end workflows—ingest to HDFS, run Hive/Spark SQL, export results—without ever altering production credentials or endpoints.

**Feature experimentation**
Try out security integrations (Kerberos, Ranger), Hive UDFs, or custom YARN schedulers locally before rolling them out cluster-wide.
