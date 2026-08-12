{
  "extends": "@react-native-community/eslint-config",
  "parser": "@babel/eslint-parser",
  "plugins": [
    [
      "module-resolver",
      {
        "root": ["./src"],
        "extensions": [".ts", ".tsx", ".js"],
        "alias": {
          "@models": "./src/models",
          "@managers": "./src/managers",
          "@components": "./src/components",
          "@screens": "./src/screens",
          "@utils": "./src/utils"
        }
      }
    ],
    "react-native-reanimated/plugin"
  ],
  "rules": {
    "react/react-in-jsx-scope": "off",
    "react/jsx-uses-react": "off",
    "no-console": "warn",
    "semi": ["warn", "always"]
  }
};
