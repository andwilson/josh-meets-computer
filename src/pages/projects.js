import React from "react";
import Link from "gatsby-link";
import Helmet from "react-helmet";
import styled from "styled-components";

import CategoryHeader from "../components/CategoryHeader";

const SLink = styled(Link)`
  text-decoration: none;
  color: #28aa55;
  &:hover {
    color: #1d7f3f;
  }
`;

class Projects extends React.Component {
  render() {
    const posts = this.props.data.allMarkdownRemark.edges;
    const categoryTitle = "Projects";
    const categoryDescription = "I am not sure exactly at what moment there was an inflection point. Perhaps it was around the time I was reading Super Intelligence. In that moment, I decided to make the creation of Artificial Intelligence my life-long single minded focus. In the subsequent years, as I traveled the world and studied data science, I started to feel what was truly relevant to people was our experiences and relationships. As a result, I refined my focus even further -- to apply A.I technology to improve human experiences and relationships. Below is a repository of projects I've tackled while taking these meandering steps towards my own A.I powered future. ";

    return (
      <div>
        <Helmet title={categoryTitle} />
        <CategoryHeader
          title={categoryTitle}
          description={categoryDescription}
          data= {this.props.data}
        />
        {posts.map(post => {
          if (
            post.node.path !== "/404/"
          ) {
            return (
              <div key={post.node.frontmatter.path}>
                <h3>
                  <SLink to={post.node.frontmatter.path}>
                    {post.node.frontmatter.title}
                  </SLink>
                </h3>
                <small>{post.node.frontmatter.date}</small>
                <p dangerouslySetInnerHTML={{ __html: post.node.excerpt }} />
              </div>
            );
          }
        })}
      </div>
    );
  }
}

export default Projects;

export const pageQuery = graphql`
  query ProjectsQuery {
    allMarkdownRemark(
      sort: { fields: [frontmatter___date], order: DESC },
      filter: {frontmatter: {category: {eq: "Projects"}}}
    ) {
      totalCount
      edges {
        node {
          excerpt(pruneLength: 300)
          frontmatter {
            path
            date(formatString: "MMMM D, YYYY")
            title
            category
          }
        }
      }
    }
    avatar: imageSharp(id: { regex: "/avatar.jpg/" }) {
      sizes(maxWidth: 500, grayscale: false) {
        ...GatsbyImageSharpSizes_tracedSVG
      }
    }
  }
`;
