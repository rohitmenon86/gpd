#define BOLD     "1"
#define UNDRL    "4"

#define FGRED      "31"
#define FGGREEN    "32"
#define FGYELLOW   "33"
#define FGBLUE     "34"
#define FGDEFAULT  "39"
#define BGRED      "41"
#define BGGREEN    "42"
#define BGYELLOW   "43"
#define BGBLUE     "44"
#define BGDEFAULT  "49"

#define MOD(x) "\033[" x "m"
#define RST   MOD("0")

#define B_MOD_RED(x)    MOD(BOLD) MOD(FGRED) x RST
#define B_MOD_GREEN(x)  MOD(BOLD) MOD(FGGREEN) x RST
#define B_MOD_YELLOW(x) MOD(BOLD) MOD(FGYELLOW) x RST
#define B_MOD_BLUE(x)   MOD(BOLD) MOD(FGBLUE) x RST
#define MOD_RED(x)      MOD(FGRED) x RST
#define MOD_GREEN(x)    MOD(FGGREEN) x RST
#define MOD_YELLOW(x)   MOD(FGYELLOW) x RST
#define MOD_BLUE(x)     MOD(FGBLUE) x RST

#define PRINT_BRED(...)  {std::cout<<B_MOD_RED(__VA_ARGS__)<<std::endl;}
#define PRINT_BGREEN(...){std::cout<<B_MOD_GREEN(__VA_ARGS__)<<std::endl;}
#define PRINT_BBLUE(...) {std::cout<<B_MOD_BLUE(__VA_ARGS__)<<std::endl;}

#define PRINT_RED(...)   {std::cout<<MOD_RED(__VA_ARGS__)<<std::endl;}
#define PRINT_GREEN(...) {std::cout<<MOD_GREEN(__VA_ARGS__)<<std::endl;}
#define PRINT_BLUE(...)  {std::cout<<MOD_BLUE(__VA_ARGS__)<<std::endl;}
#define PRINT(...)       {std::cout<<__VA_ARGS__<<std::endl;}

/* int main(int argc, char **argv)
{
    
    std::cout << "This ->" << Color::FG_RED << "word"
         << Color::FG_BLUE << "<- is red." << Color::FG_DEFAULT << std::endl;

    PRINT_RED("Test");
    //std::cout << FBLU("I'm blue.") << std::endl;
    //std::cout << BOLD(FBLU("I'm blue-bold.")) << std::endl;
    std::cout << B_MOD_RED("123") <<std::endl;

    PRINT_RED("123");
    PRINT_BRED("123");
    //PRINT_BGREEN(456);
    std::vector<int> myvect {1, 45, 46};
    PRINT_BGREEN("789"<<"258");
    PRINT_BBLUE("789"<<"258"<<"fgh");
    //PRINT_BLUE(myvect);
    //PRINT_BLUE(myvect);
    //std::cout<< Color::Code::FG_RED << myvect <<Color::Code::FG_DEFAULT<<std::endl;
    //std::cout<< myvect <<std::endl;
    return 0;
}
 */