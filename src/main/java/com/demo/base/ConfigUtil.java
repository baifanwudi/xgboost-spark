package com.demo.base;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

/**
 * Created by yn48926 on 2017/11/3.
 */
public class ConfigUtil {

    private static final Properties env;

    private static final Properties props;

    /**
     *获取配置文件
     */
    static {
        env = new Properties();
        props = new Properties();
        try {
            env.load(ConfigUtil.class.getResourceAsStream("/env.properties"));
            props.load(ConfigUtil.class.getResourceAsStream("/application.properties"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 获取key的value
     */
    public static String getEnv(String key) {
        return env.getProperty(key);
    }

    public static String getPros(String key) {
        return props.getProperty(key);
    }

    /** 返回spark的环境变量
     * @param prefix
     * @return
     */
    public static Map<String, String> getParamsWithKeyPrefix(String prefix) {
        Map<String, String> params = new HashMap<String, String>();
        env.keySet().forEach(key -> {
            String skey = key.toString();
            if (skey.startsWith(prefix)) {
                params.put(skey, ConfigUtil.getEnv(skey));
            }
        });
        return params;
    }

    public static Map<String, String> getSparkParams() {
        Map<String, String> paramsWithKeyPrefix = getParamsWithKeyPrefix("spark.");
        return paramsWithKeyPrefix;
    }

    public static Map<String, String> getHiveParams() {
        Map<String, String> paramsWithKeyPrefix = getParamsWithKeyPrefix("hive.");
        return paramsWithKeyPrefix;
    }

    public static Map<String, String> getEsParams() {
        Map<String, String> paramsWithKeyPrefix = getParamsWithKeyPrefix("es.");
        return paramsWithKeyPrefix;
    }
}