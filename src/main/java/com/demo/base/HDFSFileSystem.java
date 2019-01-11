package com.demo.base;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;

/**
 * @author allen.bai
 * hadoop 配置信息
 */
public class HDFSFileSystem {

    private static Logger logger = LoggerFactory.getLogger(HDFSFileSystem.class);

    public static FileSystem fileSystem = null;

    static {
        try {
            fileSystem = FileSystem.get(URI.create(BaseConfig.HDFS_PREFIX), new Configuration());
        } catch (IOException e) {
            e.printStackTrace();
            logger.error(e.getMessage());
        }
    }

    /**
     * 判断hdfs路径是否存在
     * @param pathList
     * @return
     * @throws IOException
     */
    public static boolean existsPath(String... pathList) throws IOException {
        for (String path : pathList) {
            if (!fileSystem.exists(new Path(path))) {
                logger.error(" the path:" + path + " is not existed");
                return false;
            }else{
                logger.warn("executing the path is : " + path);
            }
        }
        return true;
    }
}
