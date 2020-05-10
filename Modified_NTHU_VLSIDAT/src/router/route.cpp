#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cassert>

#include "parameter.h"
#include "Construct_2d_tree.h"

#include "grdb/RoutingRegion.h"
#include "grdb/parser.h"
#include "misc/filehandler.h"
#include "util/verifier.h"


void dataPreparetion(ParameterAnalyzer& ap, Builder* builder);

extern void construct_2d_tree(RoutingRegion*, const char*);
extern void Post_processing();
extern void Layer_assignment(const char*);
extern void OutputResult(const char*);

extern void PSD_output_congestion(RoutingRegion*, const char*, char *);

int main(int argc, char* argv[])
{
/*
    cout<<"======================================================="<<endl
        <<"= NTHU-Route                                          ="<<endl
        <<"= Version 2.0 is deveploped by                        ="<<endl
        <<"= Yen-Jung Chang, Yu-ting Lee, Tsung-Hsien Lee        ="<<endl
        <<"= Jhih-Rong Gao, Pei-Ci Wu                            ="<<endl
        <<"= Adviser: Ting-Chi Wang (tcwang@cs.nthu.edu.tw)      ="<<endl
        <<"= http://www.cs.nthu.edu.tw/~tcwang/nthuroute/        ="<<endl
        <<"======================================================="<<endl<<endl
        <<"======================================================="<<endl
        <<"= Running FLUTE for initial steiner tree              ="<<endl
        <<"= FLUTE is developed by Dr. Chris C. N. Chu           ="<<endl
        <<"=                       Iowa State University         ="<<endl
        <<"= http://home.eng.iastate.edu/~cnchu/                 ="<<endl
        <<"======================================================="<<endl
        <<"= Using Google Sparse Hash Map                        ="<<endl
        <<"= For internal data structure                         ="<<endl
        <<"= http://code.google.com/p/google-sparsehash/         ="<<endl
        <<"======================================================="<<endl;
*/
  
    time_t now=time(NULL);
    printf("//--------------------------------------------------------------------------------------\n");
    printf("Start timestamp: %s\n", ctime(&now));

    clock_t t0 = clock();
	ParameterAnalyzer ap(argc, argv);

	RoutingRegion* routingData = new RoutingRegion();

    dataPreparetion (ap, routingData);

    parameter_set = ap.parameter();     //Global variable: routing parameters

    routing_parameter = ap.routing_param();

    
    printf("Routing parameter:\n");
    printf("monotonic_routing_en = %d\n",routing_parameter->get_monotonic_en());
    printf("overflow_threshold = %d\n", routing_parameter->get_overflow_threshold());
    printf("simple_mode_en = %d\n", routing_parameter->get_simple_mode_en());
    printf("max_iteration_p2 = %d\n",routing_parameter->get_iteration_p2());
    printf("init_box_size_p2 = %d\n", routing_parameter->get_init_box_size_p2());
    printf("box_size_inc_p2 = %d\n", routing_parameter->get_box_size_inc_p2());
    printf("max_iteration_p3 = %d\n", routing_parameter->get_iteration_p3());
    printf("init_box_size_p3 = %d\n", routing_parameter->get_init_box_size_p3());
    printf("box_size_inc_p3 = %d\n", routing_parameter->get_box_size_inc_p3());


    pre_evaluate_congestion_cost_fp = pre_evaluate_congestion_cost_all;

    clock_t t1 = clock();
    construct_2d_tree(routingData, ap.output());
//	construct_2d_tree(routingData);
	clock_t t2 = clock();

 //   PSD_output_congestion(routingData, ap.output(),"_main.out");
 //   PSD_output_congestion(routingData, ap.output(), "_psdNTUgrStage.out");
	//psd_24: add in 2015.8.18
	printf("Entering Post Processing Stage...........\n");

	Post_processing();
//	PSD_output_congestion(routingData, ap.output(), "_final.out");
	clock_t t3 = clock();
	printf("\033[33mtime:\033[m Routing:%.2f Post Processing:%.2f Total:%.2f\n",(double)(t2-t1)/CLOCKS_PER_SEC,(double)(t3-t2)/CLOCKS_PER_SEC,(double)(t3-t1)/CLOCKS_PER_SEC);

	if(ap.caseType() == 0){
        //IBM Cases
	}else{
        //ISPD'07 Cases
        Layer_assignment(ap.output());
        clock_t t4 = clock();
        printf("\033[33mtime:\033[m Layer assignment:%.2f All:%.2f\n",(double)(t4-t3)/CLOCKS_PER_SEC,(double)(t4-t0)/CLOCKS_PER_SEC);
        
	}

  
    now=time(NULL);
    printf("Finish timestamp: %s\n", ctime(&now));
    printf("//--------------------------------------------------------------------------------------\n");

	return 0;
}

void dataPreparetion(ParameterAnalyzer& ap, Builder* builder)
{
    assert (builder != NULL);

    GRParser* parser;
	if(ap.caseType() == 0){
        parser = new Parser98(ap.input(), FileHandler::AutoFileType);
	}else{
        parser = new Parser07(ap.input(), FileHandler::AutoFileType);
	}

    parser->parse(builder);
}

